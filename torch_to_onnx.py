"""
Use this script to convert a saved PyTorch checkpoint to the ONNX format.
Argument is the path to the checkpoint and make sure the config has the values 
used to train the model (you can get them from wandb).
"""
from pathlib import Path
import argparse

import torch
import torchvision
import timm

from constants import CLASS_MAP


if __name__=="__main__":
    # Only use command line args with sweeps
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_name', type=str, help='Path to the checkpoint file.')
    args = parser.parse_args()

    cfg = {
        'input_size': 384,
        'model_name': 'resnet50',
        'model_library': 'torchvision'
    }

    # Prep checkpoint directory in repo folder
    repo_dir = Path(__file__).resolve().parent
    model_path = repo_dir / args.checkpoint_name

    # Load model to CPU (we won't run inference on GPU) and set to eval mode
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
    model.eval()
    
    # Export model by tracing with a random input of the same size as used to train
    random_input = torch.randn(1, 3, cfg['input_size'], cfg['input_size'])

    # Save the new model in the same place as the original checkpoint
    save_path = model_path.with_suffix('.onnx')
    torch.onnx.export(model, random_input, save_path, input_names=['input'], output_names=['output'])
    print(f"Converted ONNX model saved at {str(save_path)}.")
