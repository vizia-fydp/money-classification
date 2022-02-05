from pathlib import Path
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms as tf
import wandb

from dataset import MoneyDataset
from constants import CLASS_MAP

np.random.seed(0)
torch.manual_seed(0)


def train(config):
    # Get wandb setup
    run = wandb.init(project="fydp-money-classification", config=config, entity="methier")
    
    # Access all hyperparameter values through wandb.config for sweep
    cfg = wandb.config

    # Prep checkpoint directory in repo folder
    repo_dir = Path(__file__).resolve().parent
    checkpoint_dir = repo_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optimizes training speed
    torch.backends.cudnn.benchmark = True
    
    # Image size is 
    input_size = (cfg['input_size'], cfg['input_size'])
    train_transforms = tf.Compose([
        tf.RandomRotation(degrees=180, interpolation=tf.InterpolationMode.BILINEAR),
        tf.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        tf.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
        tf.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.6, hue=0.05),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = tf.Compose([
        tf.Resize(input_size),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Get indices for train/val split
    root_path = Path(cfg['root_path'])
    ds_size = len(list(root_path.glob('*/*')))
    idx = np.arange(ds_size)
    np.random.shuffle(idx)
    split_idx = int(ds_size * cfg['val_ratio'])
    val_idx = idx[:split_idx]
    train_idx = idx[split_idx:]
    
    # Create datasets and data loaders
    train_set = MoneyDataset(root_path, train_idx, train_transforms, CLASS_MAP)
    val_set = MoneyDataset(root_path, val_idx, val_transforms, CLASS_MAP)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg['batch_size'],
        pin_memory=True,
        num_workers=cfg['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg['batch_size'],
        pin_memory=True,
        num_workers=cfg['num_workers']
    )

    # Init model
    num_classes = len(CLASS_MAP)
    model = getattr(torchvision.models, cfg['model_name'])(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Freeze num_frozen layers
    if cfg['num_frozen'] > 0:
        for idx, child in enumerate(model.children()):
            # There are 4 blocks before the layers start
            if idx >= 4 + cfg['num_frozen']:
                break
            for param in child.parameters():
                param.requires_grad = False

    model = model.to(device)
    
    # Init optimizer, loss, and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['lr'], steps_per_epoch=len(train_loader), epochs=cfg['epochs'])

    best_acc = -1.0
    
    for epoch in range(cfg['epochs']):
        # Training loop
        print(f"\n=== Epoch {epoch + 1} ===")
        running_train_loss = 0.0
        train_correct = 0
        model.train()
        
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            inputs = sample_batched[0].to(device)
            labels = sample_batched[1].to(device)

            output = model(inputs)
            preds = torch.argmax(output, dim=1)
            
            loss = criterion(output, labels)
                
            running_train_loss += loss.item()
            train_correct += (preds == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
        
        # Validation loop
        running_val_loss = 0.0
        val_correct = 0
        preds_all = []
        labels_all = []
        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
                inputs = sample_batched[0].to(device)
                labels = sample_batched[1].to(device)

                output = model(inputs)
                preds = torch.argmax(output, dim=1)

                # Collect preds/labels for confusion matrix
                preds_all.append(preds)
                labels_all.append(labels)

                loss = criterion(output, labels)

                running_val_loss += loss.item()
                val_correct += (preds == labels).sum().item()

        preds_all = torch.cat(preds_all).detach().cpu().numpy()
        labels_all = torch.cat(labels_all).detach().cpu().numpy()

        val_acc = val_correct / len(val_set) * 100.0
        # Save model if better than current best
        if val_acc > best_acc:
            best_acc = val_acc
            wandb.run.summary["best_acc"] = best_acc
            torch.save(model.state_dict(), checkpoint_dir / f"{run.name}_best_model.pt")

            # Only log confusion matrix for best epoch since you can't look at history
            wandb.log({'conf_mat': wandb.plot.confusion_matrix(probs=None, y_true=labels_all, preds=preds_all, class_names=CLASS_MAP)}, commit=False)

        wandb.log({
            'train_loss': running_train_loss / len(train_loader),
            'val_loss': running_val_loss / len(val_loader),
            'train_acc': train_correct / len(train_set) * 100.0,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
   
if __name__=="__main__":
    # Only use command line args with sweeps
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=6e-4, help='max learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-3, help='weight_decay')
    parser.add_argument('--num_frozen', type=int, default=3, help='number of conv layers to freeze')
    args = parser.parse_args()

    cfg = {
        'root_path': '/home/martin/datasets/Money_Classification',
        'dataset_version': 3, # Current dataset size is 623
        'epochs': 30,
        'batch_size': 64, # chosen to give 7 roughly equal sized training batches
        'num_workers': 12,
        'val_ratio': 0.3,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'input_size': 384,
        'num_frozen': args.num_frozen,
        'model_name': 'resnet50'
    }
    train(cfg)
