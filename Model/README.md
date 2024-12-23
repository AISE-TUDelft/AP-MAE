# 
We have included a script to reproduce the trained models : train_ap_mae.py. We also include a tiny and fast demo training setup to demonstrate the process.

This script should run without issues if the package versions in requirements.txt are installed in the local python environment.

As output you should see:
- model checkpoints in ./runs/configname/SavedModels/
- performance visualizations from training in ./runs/configname/TrainImages/
- performance visualizations from testing in ./runs/configname/TestImages/



The tiny config is selected by default with line 63. This tiny config should run without issues on a local setup with any cuda compatible gpu.
```
        configname = "config_ap_mae_tiny_codegpt.py"
```
The configs to reproduce the models used in the paper can be selected as:
```
        configname = "config_ap_mae_sc2_3b.py"
        configname = "config_ap_mae_sc2_7b.py"
        configname = "config_ap_mae_sc2_15b.py"
```
Due to the combined VRAM requirements of both models and the retrieval of attention heads with output_attentions=True, the paper configurations will require A40's or comparible GPUs or significant additional VRAM management. Training these models on a relevant number of samples will require enabling the provided distributed data parallel setup (or a similar one).

Options for distributed training, wandb logging and loading models and datasets from huggingface remain included but are disabled by default for the fast demo.
