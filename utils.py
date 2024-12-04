import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss(train_losses, val_losses, output_dir=''):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    filename='loss_plot.png'
    plt.savefig(os.path.join(output_dir, filename))

def plot_results(img, pred, gt, filename, output_dir=''):
    plt.figure(figsize=(10, 10))
    
    plt.subplot(1,2,1)
    plt.imshow(img.cpu().numpy().squeeze(),cmap='gray')
    plt.imshow(pred.cpu().numpy().squeeze()[1,:,:]+pred.cpu().numpy().squeeze()[2,:,:],cmap='jet',alpha=0.5)
    plt.gca().set_title('prediction')

    plt.subplot(1,2,2)
    plt.imshow(img.cpu().numpy().squeeze(),cmap='gray')
    plt.imshow(gt.cpu().numpy().squeeze()[1,:,:]+gt.cpu().numpy().squeeze()[2,:,:],cmap='jet',alpha=0.5)
    plt.gca().set_title('gt')

    filename= filename[0].replace('.npz','.png')
    plt.savefig(os.path.join(output_dir, filename))

def get_keypoint_indices(heatmap):
    """
    Get keypoint indices from heatmap
    """

    keypoint_indices = []
    batch_size, num_keypoints, height, width = heatmap.shape
    for b in range(batch_size):
        batch_keypoints = []
        for k in range(num_keypoints):
            heatmap = heatmap_tensor[b,k,:,:]
            max_index = torch.argmax(heatmap)
            y, x = divmod(max_index.item(), width)
            batch_keypoints.append((x, y))
        keypoint_indices.append(batch_keypoints)

    return keypoint_indices