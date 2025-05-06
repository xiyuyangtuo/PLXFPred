# PLXFPred
the interpretable cross-attention network with hierarchical fusion of multimodal features improves the prediction of drug targets and affinities.

![Scheme ](https://github.com/user-attachments/assets/45c6c4a1-8b1c-461f-90ce-b8c5a1fc459c)

This repository contains the code for studying protein-ligand interactions using a bidirectional long short-term memory neural network (BILSTM) based on a graph neural network (GNN). We will use the learning content of the explainable artificial intelligence research model.

## Before you start
This code requires this esm2_t33_650M_UR50D.pt (https://github.com/facebookresearch/esm ) and ChemBERTa-zinc-base-v1 (https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1).

We suggest to run the code in a conda environment. We provide an ```environment.yml``` file that can be used to install the needed packages:

```bash
conda env create -f environment.yml
```

NOTE：If you encounter problems during the installation of the PyTorch Geometric dependencies, manually install them in the conda environment using:

```bash
pip install pytorch-gpu=2.3.0 pytorch_geometric=2.6.1 torch-scatter=2.0.8 torchmetrics=1.4.0 torchtriton=2.3.0
```

Note that the versions above are the ones found in the ```environment.yml``` file. We suggest to install such version for reproducibility, but you are encouraged to try different versions to check for compatibility!

## Dataset construction
Call ```dataset.py``` to build the training dataset, and ```dataset_pre.py``` to build the predicted dataset.
```pytorch
python dataset.py
python dataset_pre.py
```

## Train the model
We provide a ```train.py``` file to train custom models (the parameters are already built in the file).
To train your model simply run:


```pytorch
python train.py
```

## Predicting and interpreting results
Call the trained best_model_multi_task.pth model in models, run ```predict.py``` to predict the classification and regression results, and output the attention weight.And use ```wet_calm.py``` to sort the weights


```pytorch
python predict.py
python wet_calm.py
```
