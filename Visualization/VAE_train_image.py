# +
import os
import re
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

parser = argparse.ArgumentParser(description='PyTorch VAE Training Test Visualization')
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
num_epoch = 200
learning_rates = [0.0001]
weight_decays = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
optimizers = ["Adam", "AdamW"]
latent_vars = [torch.randn(4) for _ in range(3)]  # Three different fixed latent variables

# Regular expression to match the filename, allowing for scientific notation in weight decay
filename_pattern = re.compile(
    r'VAE_train_(Adam|AdamW)_(\d*\.?\d+)_([.\dEe-]+)_\d+_(\d+)_{}.pth'.format(num_epoch)
)

# Function to load model weights and generate image
def generate_image(model, latent_var, device):
    model.eval()
    with torch.no_grad():
        latent_var = latent_var.to(device)
        generated_image = model.decode(latent_var.unsqueeze(0))
        return generated_image.cpu()

# Initialize matrices to store the generated images
num_latent_vars = len(latent_vars)
image_size = (1, 28, 28)  # Assuming 28x28 grayscale images; adjust if needed
generated_images = {
    optimizer: {
        wd: np.zeros((num_latent_vars, *image_size)) for wd in weight_decays
    } for optimizer in optimizers
}
count_matrix = {
    optimizer: {
        wd: np.zeros((num_latent_vars, 1, 1, 1)) for wd in weight_decays
    } for optimizer in optimizers
}

# Load models and generate images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(z_dim=4).to(device)

for optimizer in optimizers:
    for filename in os.listdir(weights_dir):
        if filename.startswith(f'VAE_train_{optimizer}_'):
            filepath = os.path.join(weights_dir, filename)
            match = filename_pattern.match(filename)
            if match:
                # Extract information from the filename
                extracted_optimizer = match.group(1)
                lr = float(match.group(2))
                wd = float(match.group(3))  # Ensure weight decay is parsed correctly
                seed = int(match.group(4))

                print(f"Processing file: {filename}")  # Debugging print statement
                print(f"Weight decay: {wd}, Seed: {seed}")  # Debugging print statement

                if wd in weight_decays:
                    # Load model weights
                    model.load_state_dict(torch.load(filepath, map_location=device))

                    # Generate images for each latent variable
                    for idx, latent_var in enumerate(latent_vars):
                        generated_image = generate_image(model, latent_var, device)
                        generated_images[optimizer][wd][idx] += generated_image.reshape(image_size).squeeze(0).numpy()
                        count_matrix[optimizer][wd][idx] += 1

# Average the generated images
for optimizer in optimizers:
    for wd in weight_decays:
        generated_images[optimizer][wd] = np.where(
            count_matrix[optimizer][wd] > 0,
            generated_images[optimizer][wd] / count_matrix[optimizer][wd],
            np.nan
        )

# Plotting the results
print("Making plots...")
fig, axes = plt.subplots(num_latent_vars * 2, len(weight_decays), figsize=(18, 12))
fig.suptitle('Generated images for 3 latent variables')

for idx, latent_var in enumerate(latent_vars):
    for j, optimizer in enumerate(optimizers):
        for i, wd in enumerate(weight_decays):
            row = idx * 2 + j  # Calculate the row index
            image = generated_images[optimizer][wd][idx].squeeze()  # Squeeze out the channel dimension
            axes[row, i].imshow(image, cmap='gray')
            axes[row, i].axis('off')

# Setting titles for each grid
for idx, latent_var in enumerate(latent_vars):
    ax = axes[idx * 2, 0]
    ax.set_ylabel(f'Latent {idx + 1}', size='large')

# Setting titles for each column (weight decay)
for i, wd in enumerate(weight_decays):
    ax = axes[0, i]
    ax.set_title(f'wd={wd}', size='large')

# Adding legend only once, for the first row
legend_labels = [f'{optimizer}' for optimizer in optimizers]
for j, optimizer in enumerate(optimizers):
    axes[j, -1].text(30, 0, f'{optimizer}', verticalalignment='center', size='large', color='black')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('VAE_training_test_images.png')
plt.show()

