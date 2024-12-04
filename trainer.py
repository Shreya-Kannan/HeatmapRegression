from tqdm import tqdm
import argparse
from dataloading import get_loaders
from datetime import datetime
from losses import Criterion
from network import UNet2D
from utils import plot_loss
import json
import torch
import numpy
from torch import optim
import os


dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mse_weight", type=float,  default=0, help="Weight for MSE Loss")
    parser.add_argument("--kld_weight", type=float,  default=0, help="Weight for KLD Loss")
    parser.add_argument("--dice_weight", type=float,  default=0, help="Weight for Dice Loss")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--bs", default=8, type=int, help="Batch size")
    parser.add_argument("--debug_mode", default=False, type=bool, help="Debug mode for loss function")

    args = parser.parse_args()

    return args

def validate(model, criterion, val_dataloader, device, output_dir=''):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (imgs, labels, filename) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_dataloader)
    return avg_loss
            
def train(args, model, criterion, optimizer, train_dataloader, val_dataloader, device, output_dir=''):

    min_val_loss = float('inf')
    train_losses = []
    val_losses = []

    os.makedirs(f'{output_dir}/models', exist_ok=True)
    with open(f'{output_dir}/args.json', 'w') as f:
        json.dump(vars(args), f)

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=True)
        epoch_loss = 0

        for batch_idx, (imgs, keypoints, _ ) in progress_bar:
            imgs = imgs.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch+1}/{args.epochs}')
            progress_bar.set_postfix({'Train Loss': loss.item()})

        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_val_loss = validate(model, criterion, val_dataloader, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        plot_loss(train_losses, val_losses, output_dir=f'{output_dir}/plots/')
        progress_bar.set_postfix({'Train Loss': avg_train_loss, 'Val Loss': avg_val_loss})

        if avg_val_loss <  min_val_loss:
            min_val_loss = avg_val_loss
            print(f"\nValidation loss reduced. Saving model at {output_dir}/models ..")
            torch.save(model.state_dict(), f'{output_dir}/models/best_model.pth')

def main():
    args = Args()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = Criterion(kld_weight=args.kld_weight, dice_weight=args.dice_weight, mse_weight=args.mse_weight, reduction='batchmean', debug_mode=args.debug_mode)
    train_dataloader, val_dataloader = get_loaders(args.bs)
    model = UNet2D().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    train(args, model, criterion, optimizer, train_dataloader, val_dataloader, device, output_dir=f'results/{dt_string}')

    print('Training Done.')


if __name__ == "__main__":
    main()