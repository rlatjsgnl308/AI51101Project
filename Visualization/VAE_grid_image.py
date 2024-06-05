# +
import os
import re
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser(description='PyTorch VAE Grid Test Visualization')
parser.add_argument('-wdir', '--weight_dir', default="../VAE/Weight",
                    help='where is the weight pth file? (default: ../VAE/Weight)')
parser.add_argument('-mdir', '--model_dir', default="../VAE",
                    help='where is the model file? (default: ../VAE)')

args = parser.parse_args()

# Add the directory containing the VAE model to sys.path
sys.path.append(args.model_dir)

# Import the VAE model
from Model import VAE  # Make sure the file name and class name are correct

# Directory containing the model weights files
weights_dir = args.weight_dir

# Hyperparameter settings
num_epoch = 100
learning_rates = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
weight_decays = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
optimizers = ["Adam", "AdamW"]
latent_var = torch.randn(4)  # Single fixed latent variable

# Regular expression to match the filename, allowing for scientific notation in weight decay
filename_pattern = re.compile(
    r'VAE_grid_(Adam|AdamW)_([.\dEe-]+)_([.\dEe-]+)_\d+_(\d+)_{}.pth'.format(num_epoch)
)

# Function to load model weights and generate image
def generate_image(model, latent_var, device):
    model.eval()
    with torch.no_grad():
        latent_var = latent_var.to(device)
        generated_image = model.decode(latent_var.unsqueeze(0))
        return generated_image.cpu()

# Initialize matrices to store the generated images
image_size = (1, 28, 28)  # Assuming 28x28 grayscale images; adjust if needed
generated_images = {
    optimizer: np.zeros((len(learning_rates), len(weight_decays), *image_size)) for optimizer in optimizers
}
count_matrix = {
    optimizer: np.zeros((len(learning_rates), len(weight_decays))) for optimizer in optimizers
}

# Function to get indices for the matrices
def get_indices(lr, wd):
    lr_idx = learning_rates.index(lr)
    wd_idx = weight_decays.index(wd)
    return lr_idx, wd_idx

# Load models and generate images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(z_dim=4).to(device)

for optimizer in optimizers:
    for filename in os.listdir(weights_dir):
        if filename.startswith(f'VAE_grid_{optimizer}_'):
            filepath = os.path.join(weights_dir, filename)
            match = filename_pattern.match(filename)
            if match:
                # Extract information from the filename
                extracted_optimizer = match.group(1)
                lr = float(match.group(2))
                wd = float(match.group(3))  # Ensure weight decay is parsed correctly
                seed = int(match.group(4))
                
                print(f"Processing file: {filename}")  # Debugging print statement
                print(f"Learning rate: {lr}, Weight decay: {wd}")  # Debugging print statement
                if lr in learning_rates and wd in weight_decays:
                    lr_idx, wd_idx = get_indices(lr, wd)
                    
                    # Load model weights
                    model.load_state_dict(torch.load(filepath, map_location=device))

                    # Generate image for the single latent variable
                    generated_image = generate_image(model, latent_var, device)
                    generated_images[optimizer][lr_idx, wd_idx] += generated_image.reshape(image_size).squeeze(0).numpy()
                    count_matrix[optimizer][lr_idx, wd_idx] += 1

# Average the generated images
for optimizer in optimizers:
    for lr_idx in range(len(learning_rates)):
        for wd_idx in range(len(weight_decays)):
            if count_matrix[optimizer][lr_idx, wd_idx] > 0:
                generated_images[optimizer][lr_idx, wd_idx] /= count_matrix[optimizer][lr_idx, wd_idx]

# Plotting the results
print("Making plots...")
for j, optimizer in enumerate(optimizers):
    fig, axes = plt.subplots(len(weight_decays), len(learning_rates), figsize=(18, 12))
    fig.suptitle('Generated Images for Latent Variable')
    
    for lr_idx, lr in enumerate(learning_rates):
        for wd_idx, wd in enumerate(weight_decays):
            image = generated_images[optimizer][wd_idx, lr_idx].squeeze()  # Squeeze out the channel dimension
            ax = axes[wd_idx, lr_idx]
            ax.imshow(image, cmap='gray')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'VAE_grid_test_images_{optimizer}.png')
    plt.show()
