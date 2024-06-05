# +
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the stats files
stats_dir = '../VAE/Stats/'

# Hyperparameter settings
learning_rates = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
weight_decays = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
optimizers = ["Adam", "AdamW"]

# Regular expression to match the stats lines
stats_pattern = re.compile(
    r'\[epoch (\d+)\] train_total_loss : ([\d.]+), train_recon_loss : ([\d.]+), train_KL_loss : ([\d.]+), '
    r'val_total_loss : ([\d.]+), val_recon_loss : ([\d.]+), val_KL_loss : ([\d.]+)'
)

# Regular expression to match the filename
filename_pattern = re.compile(
    r'VAE_grid_(Adam|AdamW)_([.\dEe-]+)_([.\dEe-]+)_\d+_\d+\.txt'
)

# Initialize matrices for total_loss, recon_loss, KL_loss
matrix_shape = (len(learning_rates), len(weight_decays))
optimizer_matrices = {
    optimizer: {
        'total_loss': np.zeros(matrix_shape),
        'recon_loss': np.zeros(matrix_shape),
        'KL_loss': np.zeros(matrix_shape),
        'count': np.zeros(matrix_shape)
    } for optimizer in optimizers
}

# Function to get the indices for the matrix
def get_indices(lr, wd):
    lr_idx = learning_rates.index(lr)
    wd_idx = weight_decays.index(wd)
    return lr_idx, wd_idx

# Process each file in the directory
for optimizer in optimizers:
    for filename in os.listdir(stats_dir):
        if filename.startswith(f'VAE_grid_{optimizer}_'):
            filepath = os.path.join(stats_dir, filename)
            match = filename_pattern.match(filename)
            if match:
                # Extract information from the filename
                optimizer = match.group(1)
                lr = float(match.group(2))
                wd = float(match.group(3))  # Add leading zero if missing
                
                print(f"Processing file: {filename}")  # Debugging print statement
                print(f"Weight decay: {wd}")  # Debugging print statement

                if lr in learning_rates and wd in weight_decays:
                    lr_idx, wd_idx = get_indices(lr, wd)

                    # Read the file content
                    with open(filepath, 'r') as file:
                        lines = file.readlines()

                    # Find the last epoch line
                    last_epoch_stats = None
                    for line in lines:
                        stats_match = stats_pattern.match(line)
                        if stats_match:
                            last_epoch_stats = stats_match.groups()

                    if last_epoch_stats:
                        epoch, train_total_loss, train_recon_loss, train_KL_loss, val_total_loss, val_recon_loss, val_KL_loss = map(float, last_epoch_stats)
                        # Update matrices
                        optimizer_matrices[optimizer]['total_loss'][lr_idx, wd_idx] += val_total_loss
                        optimizer_matrices[optimizer]['recon_loss'][lr_idx, wd_idx] += val_recon_loss
                        optimizer_matrices[optimizer]['KL_loss'][lr_idx, wd_idx] += val_KL_loss
                        optimizer_matrices[optimizer]['count'][lr_idx, wd_idx] += 1

# Average the results
for optimizer in optimizers:
    optimizer_matrices[optimizer]['total_loss'] = np.where(
        optimizer_matrices[optimizer]['count'] > 0,
        optimizer_matrices[optimizer]['total_loss'] / optimizer_matrices[optimizer]['count'],
        np.nan
    )
    optimizer_matrices[optimizer]['recon_loss'] = np.where(
        optimizer_matrices[optimizer]['count'] > 0,
        optimizer_matrices[optimizer]['recon_loss'] / optimizer_matrices[optimizer]['count'],
        np.nan
    )
    optimizer_matrices[optimizer]['KL_loss'] = np.where(
        optimizer_matrices[optimizer]['count'] > 0,
        optimizer_matrices[optimizer]['KL_loss'] / optimizer_matrices[optimizer]['count'],
        np.nan
    )

# Plotting the matrices
print("Making plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
fig.suptitle('Validation Losses for Adam and AdamW')

# Plotting Adam
im = axes[0, 0].imshow(optimizer_matrices['Adam']['total_loss'], cmap='hot', aspect='auto')
axes[0, 0].set_title('Adam Total Loss')
axes[0, 0].set_xlabel('Weight Decay')
axes[0, 0].set_ylabel('Learning Rate')
axes[0, 0].set_xticks(range(len(weight_decays)))
axes[0, 0].set_xticklabels(weight_decays, rotation=90)
axes[0, 0].set_yticks(range(len(learning_rates)))
axes[0, 0].set_yticklabels(learning_rates)

im = axes[0, 1].imshow(optimizer_matrices['Adam']['recon_loss'], cmap='hot', aspect='auto')
axes[0, 1].set_title('Adam Reconstruction Loss')
axes[0, 1].set_xlabel('Weight Decay')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_xticks(range(len(weight_decays)))
axes[0, 1].set_xticklabels(weight_decays, rotation=90)
axes[0, 1].set_yticks(range(len(learning_rates)))
axes[0, 1].set_yticklabels(learning_rates)

im = axes[0, 2].imshow(optimizer_matrices['Adam']['KL_loss'], cmap='hot', aspect='auto')
axes[0, 2].set_title('Adam KL Loss')
axes[0, 2].set_xlabel('Weight Decay')
axes[0, 2].set_ylabel('Learning Rate')
axes[0, 2].set_xticks(range(len(weight_decays)))
axes[0, 2].set_xticklabels(weight_decays, rotation=90)
axes[0, 2].set_yticks(range(len(learning_rates)))
axes[0, 2].set_yticklabels(learning_rates)

# Plotting AdamW
im = axes[1, 0].imshow(optimizer_matrices['AdamW']['total_loss'], cmap='hot', aspect='auto')
axes[1, 0].set_title('AdamW Total Loss')
axes[1, 0].set_xlabel('Weight Decay')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_xticks(range(len(weight_decays)))
axes[1, 0].set_xticklabels(weight_decays, rotation=90)
axes[1, 0].set_yticks(range(len(learning_rates)))
axes[1, 0].set_yticklabels(learning_rates)

im = axes[1, 1].imshow(optimizer_matrices['AdamW']['recon_loss'], cmap='hot', aspect='auto')
axes[1, 1].set_title('AdamW Reconstruction Loss')
axes[1, 1].set_xlabel('Weight Decay')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_xticks(range(len(weight_decays)))
axes[1, 1].set_xticklabels(weight_decays, rotation=90)
axes[1, 1].set_yticks(range(len(learning_rates)))
axes[1, 1].set_yticklabels(learning_rates)

im = axes[1, 2].imshow(optimizer_matrices['AdamW']['KL_loss'], cmap='hot', aspect='auto')
axes[1, 2].set_title('AdamW KL Loss')
axes[1, 2].set_xlabel('Weight Decay')
axes[1, 2].set_ylabel('Learning Rate')
axes[1, 2].set_xticks(range(len(weight_decays)))
axes[1, 2].set_xticklabels(weight_decays, rotation=90)
axes[1, 2].set_yticks(range(len(learning_rates)))
axes[1, 2].set_yticklabels(learning_rates)

# Add a colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_ticks([])
cbar.set_label('Loss')

# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('VAE_grid_test.png')
plt.show()

