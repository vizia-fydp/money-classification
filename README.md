# money-classification
This repository is used to train deep learning models to classify images of US currency bills. 

## Setup
### Dataset
The dataset used is a custom dataset that was collected by the members of the FYDP team. In total, there are 706 images in the dataset which are all labeled by their bill amount (1$, 5$, 10$, 20$, 50$, or 100$) or the negative class "no bill". The dataset can be downloaded from this [Google Drive link](https://drive.google.com/drive/folders/15B_jGm-iFagokH05LWhxr89Vt52ma6WC?usp=sharing).

### Environment
A GPU is not necessarily required to run the code, but is heavily recommended. To setup the environment, first install miniconda and then create the new environment using:
```
conda env create -f environment.yml
```

## Running the code
### Training
To run the training script as is, first edit the [config dict](https://github.com/vizia-fydp/money-classification/blob/a76368166fcbd22ec249b1c98573bc9cf28ab2eb/train.py#L194) in the train.py script with your desired parameters and make sure your root directory is correct. Metric logging was done with Weights & Biases, and so you will need to run the `wandb login` command and enter your API key. If you wish to just run the training script without logging, run the `wandb disabled` command to disable logging. Then, run the training script using:
```
python train.py
```

### Sweeps
All model training was done using Weights & Biases hyperparameter sweeps to optimize the selection of the learning rate, weight decay, and number of frozen layers. You must be logged in using your wandb API key to run sweeps. To start a sweep, first run:
```
wandb sweep sweep.yaml
```

A sweep ID will be printed out in the terminal. To then start training models, run:
```
wandb agent <SWEEP_ID>
```

### Examining validation set errors
A script was also created to analyze the errors made by the model on the validation set. This script prints out some useful statistics and allows you to visualize the images that were incorrectly classified by the model. To run the script, determine the path to the checkpoint you wish to analyze and run the following:
```
python validation_stats.py <CHECKPOINT_PATH>
```

### Converting to ONNX
To reduce the size of our inference server libraries and speed up inference, we convert the trained models from PyTorch to ONNX format. To do this conversion, the torch_to_onnx script can be used. To run, ensure the config values in the script match those that were used to train the model (can be found in wandb) and then run the command:
```
python torch_to_onnx.py <CHECKPOINT_PATH>
```
