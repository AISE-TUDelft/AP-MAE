# Automated Attention Pattern Discovery at Scale in Large Language Models

This is the *reproduction package* for the paper entitled **Automated Attention Pattern Discovery at Scale in Large Language Models**

The repository is structured into two main directories:
- **Clustering**
-  **Model** 

The [**Clustering**](./Clustering) directory comprises all the code for clustering attention heads as well as the code for training the CatBoost classifier and interventions in the model inference.

The [**Model**](./Model) directory comprises all the code for building the *AP-MAE* model. This includes the model architecture and the training setup.

For further instructions on running the code, please refer to the README files in each directory. 

# Links

We list the StackLessV2 Java dataset the subset of The Heap used for training [here](https://huggingface.co/datasets/LaughingLogits/Stackless_Java_V2).

We release the AP-MAE model collection [here](https://huggingface.co/collections/LaughingLogits/ap-mae-models-66b27a73536bb1306d55c4c4).
