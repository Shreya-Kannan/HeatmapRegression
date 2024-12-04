from tqdm import tqdm
import argparse
from dataloading import get_loaders
from datetime import datetime
from network import UNet2D
from utils import plot_results
from losses import Criterion
import json
import torch
import numpy as np
from torch import optim
import os

def Args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=1, type=int, help="Batch size")
    parser.add_argument("--model_dir", type=str, help="Directory to saved model")
    parser.add_argument("--dataset_path", type=str, help="Directory to dataset for inference") #need to finish this
    parser.add_argument("--save_file", default=False, type=bool, help="Save npz of prediction")
    parser.add_argument("--mse_weight", type=float,  default=0, help="Weight for MSE Loss")
    parser.add_argument("--kld_weight", type=float,  default=0, help="Weight for KLD Loss")
    parser.add_argument("--dice_weight", type=float,  default=0, help="Weight for Dice Loss")

    args = parser.parse_args()

    return args

def test(model, test_dataloader, criterion, device, output_dir='', save_file=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (imgs, labels, filename) in enumerate(test_dataloader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            outputs = torch.softmax(outputs, dim=1)
            print(f'Saving the prediction for patient {filename[0]}')

            if save_file:
                np.savez(f'{output_dir}/{filename[0]}', pred = outputs.cpu().numpy(), img=imgs.cpu().numpy(), loss=loss.item())
            
            plot_results(imgs, outputs, labels, filename, output_dir)

    avg_loss = total_loss / len(test_dataloader)
    print(f'Average loss for test set: {avg_loss}\n')


def main():
    args = Args()
    os.makedirs(f'{args.model_dir}/predictions', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, test_dataloader = get_loaders(args.bs, train=False)

    model = UNet2D().to(device)
    criterion = Criterion(kld_weight=args.kld_weight, dice_weight=args.dice_weight, mse_weight=args.mse_weight, reduction='batchmean', debug_mode=False)
    model_path = f'{args.model_dir}/best_model.pth'
    print(f'Loading model from {model_path}')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test(model, test_dataloader, criterion, device, output_dir = f'{args.model_dir}/predictions', save_file=args.save_file)

    print('Inference Done.')


if __name__ == "__main__":
    main()