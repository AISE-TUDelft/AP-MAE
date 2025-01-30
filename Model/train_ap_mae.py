"""
This is a demo training script for training PA-MAE models on attention heads

We recently observed issues with the multithreaded datasets package download & 
split generation functions. We recommend using our download_dataset.py first.

By default this script will simply run off a local config and 1 gpu.
This should run without issues if the package versions in requirements.txt 
are installed in the local python environment, and the dataset splits have been 
downloaded and cached with download_dataset.py

As output you should see:
    model checkpoints in ./runs/configname/SavedModels/
    performance visualizations from training in ./runs/configname/TrainImages/
    performance visualizations from testing in ./runs/configname/TestImages/

We have provided a fast, tiny demonstration configuration file which should 
run on any modern cuda compitable gpu. It is selected by default with line 63:
        configname = "config_ap_mae_tiny_codegpt"

The configs to reproduce the models used in the paper can be selected as:
        configname = "config_ap_mae_sc2_3b"
        configname = "config_ap_mae_sc2_7b"
        configname = "config_ap_mae_sc2_15b"
Due to the combined VRAM requirements of both models and the retrieval 
of attention heads with output_attentions=True, these configurations will 
require A40's or comparible GPUs (or significant additional VRAM management).
Training the models on a relevant number of samples will require enabling the
provided distributed data parallel setup (or a similar one).

Options for distributed training, wandb logging and loading models and datasets from huggingface remain included but are disabled by default for the fast demo.
"""

import os
from DataUtil.Common import read_file, IndexableDict

# optionally connect to huggingface to load private models and datasets
use_huggingface_login = False
huggingface_token_filename = './YOUR_HUGGINGFACE_TOKEN_FILE.txt'
# required for local datasets and models if not using huggingface

# optionally connect and log results to wandb, requires a wandb keyfile
use_wandb = False
wandb_key_filename = './YOUR_WANDB_KEY_FILE.txt'
# required if using wandb without a sweep configuration
wandb_project_name = 'YOUR_WANDB_PROJECT_NAME'
wandb_entity_name = 'YOUR_WANDB_ENTITY_NAME'

# optionally disable distributed data parallel torch backend for single gpu run
# backend might not be compatible with Windows or certain distros
single_gpu_disable_ddp_backend = True

if use_wandb:
    import wandb
    wandb_key = read_file(wandb_key_filename)
    wandb.login(key=wandb_key)

# option to pull config from environment var (for scheduling slurm scripts)
try:
    configname = os.environ['RUNCONFIG']
except KeyError:
    print("RUNCONFIG environment variable not set, using manual setting")
    # choose a run config manually here
    configname = "config_ap_mae_tiny_codegpt"
print("using config name:", configname)

# retrieve config from configname.py
runconfig = getattr(__import__(configname, fromlist=[configname]), configname)
if use_wandb:
    wandb.init(config=runconfig, project=wandb_project_name, entity=wandb_entity_name)
    run_config = wandb.config
else:
    run_config = IndexableDict(runconfig)

print("use_wandb:", use_wandb, "run_config:", run_config)

# Location for local huggingface cache files.
# in order to work these vars need to be set BEFORE huggingface imports
os.environ['HF_HOME'] = run_config.hf_home

if use_huggingface_login:
    from huggingface_hub import login
    auth_token = read_file(huggingface_token_filename)
    login(auth_token)




from transformers import AutoModelForCausalLM, AutoTokenizer
from ap_mae import APMAE, APMAEConfig
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange
from DataUtil.DDPDataLoader import IterableAttentionDataset
from DataUtil.Scalers import select_scaler
from tqdm import tqdm
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches
import matplotlib.colors as pltcolors
import random

def img_from_patches(config, patched_masked_image, masked_indices, gpu, image_blank_value=0, base_image=None):
    """
    Converts masked patchified images back to a regular image shape.

    example:
        - original image = 32x32, patch size = 8
        - patched image = 16 patches of 8x8
        - mask selection = 5 patches, ids: 0,8,9,11,13\n
        input :
            patched_masked_image = tensor of shape [1,5,64]\n
            masked_indices = tensor [0,8,9,11,13]\n
        output :
            image = tensor of shape [1,64,64] with patches filled in as if the image were a 4x4 grid of 16x16 patches and the selected locations were the 0,8,9,11,13 indices.
    
    base_image : if specified, patches are filled in on top of this image.
    image_blank_value : if base_image is None, this gets used as background color for missing patches.
    """
    image_size = config.max_length
    patch_size = config.patch_size
    # width of grid of patches : image_size=8, patch_size=4 -> patch_dim=2
    patch_dim = image_size//patch_size

    # recreate the full patch grid, not just selected patches
    if base_image is None:
        # create a blank base canvas in a shape with all patches
        base_canvas = torch.ones((patch_dim**2, patch_size*patch_size), device = gpu)*image_blank_value
    else:
        base_image = base_image.to(gpu)
        base_canvas = rearrange(base_image, '(h p1) (w p2) -> (h w) (p1 p2)', p1 = patch_size, p2 = patch_size)
    # fill in the masked patches at correct indices
    base_canvas[masked_indices] = patched_masked_image

    # convert patch grid back to image dimensions
    # reshape to provide dimension info for einsum
    base_canvas = base_canvas.reshape(patch_dim, patch_dim, patch_size, patch_size)
    # need to convert patch rows&cols into image grid rows&&cols
    base_canvas = torch.einsum("yxhw->yhxw", base_canvas)
    image = base_canvas.reshape(image_size, image_size)
    # grab first from batch and send to cpu for matplot
    return image

def visualize_sample(config, model, pixel_values, epoch, samples_seen, selected_head, visualizations_table, gpu, output_dir, save_data=False):
    with torch.inference_mode():
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        output_name = f"{output_dir}epoch{str(epoch)}_i{str(samples_seen)}_h{str(selected_head)}"
        image_size = config.max_length
        patch_size = config.patch_size
        grid_size = image_size//patch_size
        pixel_values = pixel_values.to(gpu)

        loss, pred_pixel_values, masked_indices, masked_patches, unmasked_indices = model(pixel_values, visualizing=True)

        # for future analysis / reconstructing images
        if save_data:
            torch.save(pixel_values, f"{output_name}_pixel_values")
            torch.save(loss, f"{output_name}_loss")
            torch.save(pred_pixel_values, f"{output_name}_pred_pixel_values")
            torch.save(masked_indices, f"{output_name}_masked_indices")
            torch.save(unmasked_indices, f"{output_name}_unmasked_indices")
        
        def show_image(axes, image, title=None, hide_top_right=False, hide_patches=None, title_x=None, title_y=None, subtitle=None, subtitle_x=None, subtitle_y=None):
            """
            vmin/vmax : optional cutoff values for input data
            """
            if config.visualize_norm == "log_normalize":
                lognorm_vmin = None
                lognorm_vmax = None
                norm = pltcolors.LogNorm(vmin=lognorm_vmin,vmax=lognorm_vmax)
                cmap = mpl.colormaps['viridis']
                cmap.set_bad(cmap.colors[0]) # set undefined log 0 = 0
                vmin=None
                vmax=None
            else:
                norm = None
                # cmap = 'gray'     # test grayscale image
                cmap = None
                vmin=None
                vmax=None
            img = axes.imshow(
                image,
                interpolation='none', # use true data pixel values
                cmap=cmap,
                norm = norm,
                vmin=vmin,
                vmax=vmax,
            )
            # optional logic to hide patches or show a grid:
            def _hide_patches(indices, linewidth=0):
                for p in indices:
                    col,row = divmod(p.item(), grid_size)
                    rect = pltpatches.Rectangle((row*patch_size-0.5, col*patch_size-0.5), patch_size, patch_size, linewidth=0, color='w', fill=True)
                    axes.add_patch(rect)
                    rect = pltpatches.Rectangle((row*patch_size-0.5, col*patch_size-0.5), patch_size, patch_size, linewidth=linewidth, edgecolor='k', linestyle='-', facecolor='k', fill=False)
                    axes.add_patch(rect)
            if hide_top_right:
                top_right_indices = torch.triu_indices(grid_size, grid_size, offset=1)
                id_matrix = torch.arange(0,grid_size**2).reshape(grid_size, grid_size)
                top_right_indices = id_matrix[top_right_indices[0], top_right_indices[1]]
                _hide_patches(top_right_indices, linewidth=1)
            if hide_patches is not None:
                _hide_patches(hide_patches, linewidth=1)
            
            # add title / subtitle
            if title_x is None:
                title_x = 0.5
            if title_y is None:
                title_y = 1.02
            if subtitle_x is None:
                subtitle_x = 0.5
            if subtitle_y is None:
                subtitle_y = -0.02
            if title is not None:
                axes.text(title_x,title_y,title,horizontalalignment='center',verticalalignment='bottom', transform=axes.transAxes, fontsize=14)
            if subtitle is not None:
                axes.text(subtitle_x,subtitle_y,subtitle,horizontalalignment='center',verticalalignment='top', transform=axes.transAxes, fontsize=14)
            # disable pixel location axes&labels for subplots
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            
            return img
        
        # set figure for AAAI paper
        width = 7        # colwidth : 3.325
        height = 2.5       # height for colwidth : ~1
        plt.rcParams['figure.figsize'] = [width,height]
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['font.size'] = 11
        plt.rcParams['figure.dpi'] = 300 

        fig, axes = plt.subplots(nrows=1, ncols=4)
        # tight grouping, with some space to the right for one colorbar
        fig.subplots_adjust(left=0,bottom=0,right=0.91,top=1, wspace=0.05, hspace=0)

        original_image = pixel_values.squeeze(dim=1)[selected_head].cpu()
        selected_patches = img_from_patches(
            config, 
            torch.zeros(len(masked_indices[selected_head]),config.patch_size**2).to(gpu), 
            masked_indices[selected_head], 
            gpu, 
            base_image = original_image, 
        ).cpu()
        predicted_patches = img_from_patches(
            config, 
            pred_pixel_values[selected_head], 
            masked_indices[selected_head], 
            gpu, 
        ).cpu()
        reconstructed = img_from_patches(
            config, 
            pred_pixel_values[selected_head], 
            masked_indices[selected_head], 
            gpu, 
            base_image = original_image, 
        ).cpu()
        
        show_image(axes[0], original_image, "original attention")
        show_image(axes[1], selected_patches, "unmasked input", hide_top_right=True, hide_patches=masked_indices[selected_head])
        show_image(axes[2], predicted_patches, f"predicted output", hide_top_right=True, hide_patches=unmasked_indices[selected_head],subtitle=f"loss:{loss:.2e}")
        img = show_image(axes[3], reconstructed, f"combined reconstruction", hide_top_right=True, title_x=0.6)

        # add colorbar, using colordata scale from last subplot
        cax = plt.axes((0.92, 0.115, 0.02, 0.77))
        plt.colorbar(img, cax=cax)
        plt.savefig(f"{output_name}.png", dpi=300)

        if visualizations_table is not None:
            visualizations_table.add_data(
                samples_seen,
                selected_head, 
                wandb.Image(output_name),
                loss
            )
        plt.close()
        return loss

def train(config, model, train_loader, scaler, optimizer, scheduler, epoch, gpu, visualizations_table, output_dir, rank=0):
    t1 = datetime.now()
    model.train()
    with tqdm(train_loader, unit = 'Samples', total = len(train_loader), disable=False) as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (attentions, query) in enumerate(tepoch):
            if use_wandb:
                wandb.log({"Learning Rate": scheduler.get_last_lr()[0]})
            optimizer.zero_grad()
            scaled = scaler(attentions, config)
            loss = model(scaled)
            loss.backward()

            # visualize before optimizer step = not yet trained on
            if (batch_idx % config.visualize_frequency == 0) or (len(tepoch) == 1):
                # random visualize one head from current batch
                heads_seen = batch_idx*len(scaled)
                model.eval()
                selected_head = random.randint(0,len(scaled)-1)
                visualize_sample(config, model, scaled, epoch, heads_seen, selected_head, visualizations_table, gpu, output_dir=f"{output_dir}/TrainImages/")
                model.train()
            
            optimizer.step()
            scheduler.step()
            tepoch.set_postfix(PREV_loss = loss.item())
            if use_wandb:
                wandb.log({f"{query} Batch Loss": loss.item()})
            
            if rank == 0:
                if ((batch_idx+1) % config.save_model_frequency == 0) or (len(tepoch) == 1):
                    t_save = datetime.now()
                    print(f"{t_save} saving model ... ", flush=True)
                    os.makedirs(os.path.dirname(f'{output_dir}/SavedModels/'), exist_ok=True)
                    if ddp_backend_enabled:
                        model.module.save_pretrained(f'{output_dir}/SavedModels/Model_e{epoch}_{batch_idx+1}')
                    else:
                        model.save_pretrained(f'{output_dir}/SavedModels/Model_e{epoch}_{batch_idx+1}')
                    print(f"saving done ({(datetime.now() - t_save).total_seconds()}s)", flush=True)
    print("train time: ", (datetime.now() - t1).total_seconds())

def test(config, model, test_loader, scaler, gpu, output_dir):
    t1 = datetime.now()
    model.eval()
    with torch.inference_mode():
        val_loss = 0
        val_batches = len(test_loader)
        for batch_idx, (attentions, query) in enumerate(test_loader):
            scaled = scaler(attentions, config)
            selected_head = random.randint(0,len(scaled)-1)
            loss = visualize_sample(config, model, scaled, 'test', batch_idx, selected_head, visualizations_table=None, gpu=gpu, output_dir=f'{output_dir}/TestImages/')
            if use_wandb:
                wandb.log({f"{query} Batch Validation Loss": loss.item()})
            val_loss += loss.item()
        val_loss /= val_batches
        if use_wandb:
            wandb.log({'validation_loss': val_loss})
    print("test time: ", (datetime.now() - t1).total_seconds())

ddp_backend_enabled = None
def main():
    config = APMAEConfig(**run_config)
    if use_wandb:
        output_dir = f'./runs/{wandb.run.id}/'
    
        # Visualization table for logging images to wandb
        data = []
        columns = ['samples_seen', 'selected_head', 'image', 'loss']
        visualizations_table = wandb.Table(data=data, columns=columns)
    else:
        output_dir = f'./runs/{configname}/'
        visualizations_table = None

    # for reproducibility
    # AND FOR DISTRIBUTED TRAINING OF NEW MODELS: SAME WEIGHTS INITIALIZATION
    torch.manual_seed(config.initial_seed)
    random.seed(config.initial_seed)
    
    print("available cuda devices:", flush=True)
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        print("cuda", i, " : ", torch.cuda.get_device_properties(i).name, flush=True)
    try:
        if os.environ["CLUSTER"] == "CLUSTER_NAME_1":
            world_size     = int(os.environ["SLURM_NTASKS"])
            rank           = int(os.environ["SLURM_PROCID"])
            gpus_per_task = int(int(os.environ["SLURM_GPUS_PER_TASK"].split(":")[-1]))
        if os.environ["CLUSTER"] == "CLUSTER_NAME_2":
            world_size     = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_NTASKS_PER_NODE"])
            rank           = int(os.environ["SLURM_PROCID"])
            gpus_per_task = int(os.environ["SLURM_GPUS_ON_NODE"]) // int(os.environ["SLURM_NTASKS_PER_NODE"])
        ddp_backend_enabled = True
    except:
        print("cluster name not found, assuming local config with 1 gpu")
        world_size = 1
        rank = 0
        gpus_per_task = 1
        if single_gpu_disable_ddp_backend:
            ddp_backend_enabled = False
    
    print(f"rank = {rank}, world_size = {world_size}, gpu's visible = {device_count}, gpu's per task = {gpus_per_task}")

    if gpus_per_task > device_count:
        print(f"rank {rank} : error resource mismatch gpus_per_node={gpus_per_task} torch.cuda.device_count()={device_count}", flush=True)
        exit()
    
    if gpus_per_task < device_count:
        gpu = rank - device_count * (rank // device_count)
        print(f"can see more than one gpu, some tasks on same node, cuda:{gpu} chosen for rank {rank}", flush=True)
    else:
        gpu = 0
        print(f"cuda:{gpu} chosen for rank {rank}", flush=True)

    print("ddp_backend_enabled:", ddp_backend_enabled)
    if ddp_backend_enabled:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    eff_batch_size = config.train_batch_size * world_size
    eff_learning_rate = config.base_learning_rate * eff_batch_size/50
    print("base lr: %.2e" % (config.base_learning_rate))
    print("effective batch size: %d" % eff_batch_size)
    print("effective batch size adjusted lr: %.2e" % eff_learning_rate)

    with torch.inference_mode():
        tokenizer = AutoTokenizer.from_pretrained(config.target_model_name)
        # device map model parallelism if set and model supports it
        if config.target_model_device == 'device_map':
            # fix wandb forcing dict string keys
            if config.target_model_memory_limits == None:
                target_model_memory_limits = None
            else:
                target_model_memory_limits = {}
                for k,v in config.target_model_memory_limits.items():
                    if v == 0:
                        continue
                    if k == 'cpu':
                        target_model_memory_limits[k] = v
                    else:
                        target_model_memory_limits[int(k)] = v
            target_model = AutoModelForCausalLM.from_pretrained(
                config.target_model_name,
                device_map = config.target_model_device_map,
                offload_folder = config.target_model_offload_folder,
                max_memory = target_model_memory_limits,
            )
        else:
            target_model = AutoModelForCausalLM.from_pretrained(
                config.target_model_name, 
                # use accelerate to immediately load on gpu, no cpu mem used.
                device_map = gpu, 
            )
        target_model.eval()
    
    train_loader = IterableAttentionDataset(
        config = config, 
        dataset_location = config.dataset_location, 
        dataset_split = config.dataset_train_split, 
        max_batches = config.train_batches, 
        min_length = config.min_length, 
        max_length = config.max_length, 
        queries = config.queries, 
        lang = config.lang, 
        correct_only = config.correct_only, 
        target_model_name = config.target_model_name, 
        target_model_device = gpu, 
        num_proc = config.iter_loader_workers, 
        reset_after_iter = True, 
        equal_query_quantities = True, 
        rank = rank, 
        world_size = world_size, 
        batch_size = config.train_batch_size, 
        tokenizer = tokenizer, 
        target_model = target_model, 
        head_selection_strategy = config.train_head_selection_strategy, 
    )
    test_loader = IterableAttentionDataset(
        config = config, 
        dataset_location = config.dataset_location, 
        dataset_split = config.dataset_test_split, 
        max_batches = config.val_batches, 
        min_length = config.min_length, 
        max_length = config.max_length, 
        queries = config.queries, 
        lang = config.lang, 
        correct_only = config.correct_only, 
        target_model_name = config.target_model_name, 
        target_model_device = gpu, 
        num_proc = config.iter_loader_workers, 
        reset_after_iter = True, 
        equal_query_quantities = True, 
        rank = rank, 
        world_size = world_size, 
        batch_size=config.test_batch_size, 
        tokenizer = tokenizer, 
        target_model = target_model, 
        head_selection_strategy = config.test_head_selection_strategy, 
    )
    
    if config.ap_mae_preload_name is not None:
        model = APMAE.from_pretrained(
            pretrained_model_name_or_path=config.ap_mae_preload_name
        ).to(gpu)
        print(f"continuing training of old run: {config.ap_mae_preload_name}")
    else:
        model = APMAE(config).to(gpu)
        print("training new model from scratch.")
    if ddp_backend_enabled:
        model = DDP(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = eff_learning_rate, 
        betas=[0.9, 0.95], 
        weight_decay=0.05, 
    )
    # warm restarts after loader length = after each epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        len(train_loader), 
    )
    scaler = select_scaler(config)

    for epoch in range(1, config.max_epochs + 1):
        train(config, model, train_loader, scaler, optimizer, scheduler, epoch, gpu, visualizations_table, output_dir = output_dir, rank = rank)
    test(config, model, test_loader, scaler, gpu, output_dir = output_dir)

    if use_wandb:
        print("logging visualizations to wandb ... ", flush=True)
        # log the visualizations tableto wandb
        wandb.run.log({"visualizations" : visualizations_table})

    if ddp_backend_enabled:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
