# Image Classification

This project provides a framework for training a image classification model using PyTorch. The framework supports logging with Weights and Biases (WandB).


## To Use To:

- Train classification for Damage Assesment

## Directory Structure

- `main.py`: Main script to run the data analysis and training.
- `config.py`: Configuration file for hyperparameters and paths.
- `data/`: Folder containing the dataset loading script.
- `models/`: Folder containing the model definition script.
- `analysis/`: Folder containing the data analysis script.
- `training/`: Folder containing the training script.
- `inference/`: Folder containing the inference script.
- `utils/`: Folder containing utility functions.
- `logs/`: Folder containing the log files.

## Getting Started

```bash
   conda create -n multimodal_env python=3.8
   conda activate multimodal_env
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   conda install -c conda-forge openssl
   conda install -c conda-forge libssl1.0.0
   conda install -c huggingface transformers
   conda install pandas seaborn matplotlib scikit-learn
   pip install wandb tqdm
```

1. **Data Analysis**: Run `main.py` to perform data analysis and log the results to WandB.

```bash
   python main.py
```
