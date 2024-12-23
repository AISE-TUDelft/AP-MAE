"""
This configuration trains a ap_mae model on attention head images 
from inference inputs of 256 token length to the StarCoder2-3B model.
"""
config_ap_mae_sc2_3b = {
    # ==== model loading options ====
    # 'target_model_name' : 'bigcode/starcoderbase',
    'target_model_name' : 'bigcode/starcoder2-3b',
    # 'target_model_name' : 'bigcode/starcoder2-7b',
    # 'target_model_name' : 'bigcode/starcoder2-15b',
    # 'target_model_name': 'bigcode/starcoderbase-1b',
    # 'target_model_name' : 'microsoft/CodeGPT-small-java-adaptedGPT2',

    # specify single device to load target model on
    'target_model_device' : 'cuda:0',
    # 'target_model_device' : 'cpu',

    # alternatively : specify device map to split target model across gpus
    # this can help to fit the target model on smaller GPU's
    # 'target_model_device' : 'device_map',
    # additional settings required for device mapping
    # 'target_model_device_map' : 'auto',
    # 'target_model_offload_folder' : './offload',
    # 'target_model_memory_limits' : None,
    # 'target_model_memory_limits' : {'cpu': '10GiB', 0: '3GiB'},

    # optional simple device map to fit trained model on smaller GPU's
    # 'encoder_device' : 'cpu',
    'encoder_device' : 'cuda:0',
    # 'decoder_device' : 'cpu',
    'decoder_device' : 'cuda:0',


    # ==== data selection options ====
    'dataset_location' : 'LaughingLogits/Stackless_Java_V2', 
    'dataset_name' : 'Stackless_Java_V2', 
    'dataset_train_split' : 'train',
    'dataset_test_split' : 'test',

    # note for distributed setup: the dataset_train_split gets split across 
    # process ranks for a 100% guarantee of training on distinct samples. 
    #   Example: world size 8, 100k initial dataset
    #     rank 0 will train on the subset document sample ids 0-12500

    # match n_cpu_threads available (or slightly less, for other tasks)
    'iter_loader_workers' : 8,

    # how often do we repeat the training process (re-shuffle dataset split, reset learning rate scheduler)
    'max_epochs' : 1,
    # How many batches of attention heads will each rank train on per epoch.
    # Dataset is iterated over for inference until the desired number of attention heads is reached. The [document -> token inference -> attention heads -> batch] relation depends on the chosen query and attention selection strategy.
    'train_batches' : 150000,
    # we validate only at the end after all epochs + visualize all its results.
    'val_batches' : 3840,
    # how many attention head 'images' at once to vision transformer
    # for ddp: this is the local batch size, effective = local*world_size
    'train_batch_size' : 60,
    'test_batch_size' : 1,
    # document language selection for input tokens to train on
    'lang' : 'java',
    # query type selection for input tokens to train on
    'queries': ['random'],
    # 'queries' : ['identifiers', 'numeric literals', 'string literals', 'boolean literals', 'function call', 'function name'],
    # 'queries': ['identifiers', 'string literals'],
    # attention scaling method selection for effective vision model training
    # 'attention_scaler': None,
    # 'attention_scaler': 'log',
    # 'attention_scaler': 'log_standardize',
    'attention_scaler': 'log_normalize',
    # specify which attention heads to take from inference samples
    'train_head_selection_strategy': ('layerwise', 0.25),
    # 'train_head_selection_strategy': ('all'),
    # 'test_head_selection_strategy': ('layerwise', 0.25),
    'test_head_selection_strategy': ('all'),
    # max input seq token length == nxn attention head image size
    'max_length' : 256,
    'min_length' : 256,
    # whether to only train on inferences that had a correct prediction
    'correct_only' : True,
    # visualize prediction performance of the trained model every ... batches
    'visualize_frequency' : 2000,
    # should the pixel values in the visualization be scaled?
    # 'visualize_norm': 'log_normalize', 
    'visualize_norm': None, 
    # save trained model every ... batches
    'save_model_frequency' : 15000,




    # ==== config for the trained ap_mae model ====
    # optional : specify a folder to load saved pretrained weights
    #  - new model config must match the previously pretrained model
    #  - assumes folder contains both encoder and decoder weight components.
    'ap_mae_preload_name' : None,
    # 'ap_mae_preload_name' : './runs/config_ap_mae_tiny_codegpt/SavedModels/Model_e1_10000/',

    'patch_size' : 32,
    'mask_ratio' : 0.5,

    'encoder_dim' : 512,
    'encoder_layers' : 24,
    'encoder_heads' : 16,
    'encoder_dim_head' : 64,
    'encoder_mlp_dim' :2048,

    'decoder_dim' : 512,
    'decoder_layers' : 8,
    'decoder_heads' : 8,
    'decoder_dim_head' : 64,
    'decoder_mlp_dim' :2048,

    'base_learning_rate' : 1.5e-4,

    # seed for rng, to ensure reproducibility and to initialize model in
    # the same state across nodes for ddp training.
    'initial_seed' : 45, 
    'dataset_split_seed' : 42, 
    # A new seed should be picked when using ap_mae_preload_name to continue training a model ! Otherwise the dataloader will yield the same samples in the same order as already trained on during the previous training run.



    # huggingface cache location
    'hf_home' : './huggingface',
}
