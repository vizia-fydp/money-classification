"""
This script runs inference on the validation set and outputs various stats as well as displays incorrect predictions.
"""
from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms as tf
import timm
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from dataset import MoneyDataset, balanced_split
from constants import CLASS_MAP
import utils


np.random.seed(0)
torch.manual_seed(0)


def eval(cfg, checkpoint_name):
    # Prep checkpoint directory in repo folder
    repo_dir = Path(__file__).resolve().parent
    model_path = repo_dir / checkpoint_name

    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Image size is 
    input_size = (cfg['input_size'], cfg['input_size'])
    val_transforms = tf.Compose([
        tf.Resize(input_size),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get file lists for train/val split
    root_path = Path(cfg['root_path'])
    _, val_list = balanced_split(root_path, cfg['val_ratio'], CLASS_MAP)
    
    # Create datasets and data loaders
    val_set = MoneyDataset(root_path, val_list, val_transforms, CLASS_MAP)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers']
    )

    if cfg['model_library'] == 'torchvision':
        # Init model
        model = getattr(torchvision.models, cfg['model_name'])()
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_MAP))
    elif cfg['model_library'] == 'timm':
        # Init model
        model = timm.create_model(cfg['model_name'], num_classes=len(CLASS_MAP))
    else:
        raise ValueError("Invalid model_library config parameter.")

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Validation loop
    val_correct = 0
    preds_all = []
    labels_all = []
    incorrect_dict = {"images": [], "labels": [], "preds": []}
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
            inputs = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)

            output = model(inputs)
            preds = torch.argmax(output, dim=1)

            # Collect preds/labels for confusion matrix
            preds_all.append(preds)
            labels_all.append(labels)

            val_correct += (preds == labels).sum().item()

            # Keep track of images, labels, and preds for all incorrect preds
            incorrect_idx = (preds != labels).nonzero()
            for idx in incorrect_idx:
                i = idx.item()
                incorrect_dict['images'].append(utils.tensor_to_np(inputs[i].detach().cpu()))
                incorrect_dict['labels'].append(CLASS_MAP[labels[i].item()])
                incorrect_dict['preds'].append(CLASS_MAP[preds[i].item()])

    val_acc = val_correct / len(val_set) * 100.0

    print(f"Accuracy: {val_acc}")
    print(f"Number of incorrect examples: {len(val_set) - val_correct}")

    preds_all = torch.cat(preds_all).detach().cpu().numpy()
    labels_all = torch.cat(labels_all).detach().cpu().numpy()

    print("Displaying confusion matrix...")
    ConfusionMatrixDisplay.from_predictions(labels_all, preds_all, labels=list(range(len(CLASS_MAP))), display_labels=CLASS_MAP)
    plt.show()

    print("Showing incorrrect predictions...")
    for i in range(len(incorrect_dict['labels'])):
        label = incorrect_dict['labels'][i]
        pred = incorrect_dict['preds'][i]
        plt.imshow(incorrect_dict['images'][i])
        plt.title(f"Label: {label}, Prediction: {pred}")
        plt.show()
   
if __name__=="__main__":
    # Only use command line args with sweeps
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_name', type=str, help='max learning rate')
    args = parser.parse_args()

    cfg = {
        'root_path': '/home/martin/datasets/Money_Classification',
        'dataset_version': 4, # Current dataset size is 535 train and 173 val
        'batch_size': 27, # Max size that fits with nfroz=1
        'num_workers': 12,
        'val_ratio': 0.25,
        'input_size': 384,
        'model_name': 'resnet50',
        'model_library': 'torchvision'
    }
    eval(cfg, args.checkpoint_name)
