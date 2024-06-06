# +
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the stats files
stats_dir = '../VGG/Stats/'

# Hyperparameter settings
learning_rates = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
weight_decays = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
optimizers = ["Adam", "AdamW"]

# Regular expression to match the stats lines
stats_pattern = re.compile(
    r'\[epoch (\d+)\] train_loss : ([\d.]+), train accu : ([\d.]+) , val_loss : ([\d.]+), val accu : ([\d.]+)'
)


# Regular expression to match the filename
filename_pattern = re.compile(
    r'VGG_grid_(Adam|AdamW)_([.\dEe-]+)_([.\dEe-]+)_\d+_\d+\.txt'
)

# Initialize matrices for total_loss, recon_loss, KL_loss
matrix_shape = (len(learning_rates), len(weight_decays))
optimizer_matrices = {
    optimizer: {
        'loss': np.zeros(matrix_shape),
        'accuracy': np.zeros(matrix_shape),
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
        if filename.startswith(f'VGG_grid_{optimizer}_'):
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
                        epoch, train_loss, train_accuracy, test_loss, test_accuracy = map(float, last_epoch_stats)
                        # Update matrices
                        optimizer_matrices[optimizer]['loss'][lr_idx, wd_idx] += test_loss
                        optimizer_matrices[optimizer]['accuracy'][lr_idx, wd_idx] += test_accuracy
                        optimizer_matrices[optimizer]['count'][lr_idx, wd_idx] += 1

# Average the results
for optimizer in optimizers:
    optimizer_matrices[optimizer]['loss'] = np.where(
        optimizer_matrices[optimizer]['count'] > 0,
        optimizer_matrices[optimizer]['loss'] / optimizer_matrices[optimizer]['count'],
        np.nan
    )
    optimizer_matrices[optimizer]['accuracy'] = np.where(
        optimizer_matrices[optimizer]['count'] > 0,
        optimizer_matrices[optimizer]['accuracy'] / optimizer_matrices[optimizer]['count'],
        np.nan
    )

# Plotting the matrices
print("Making plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 15), constrained_layout=True)
fig.suptitle('Test Stats for Adam and AdamW')

# Plotting Adam
im = axes[0, 0].imshow(optimizer_matrices['Adam']['loss'], cmap='hot', aspect='auto')
axes[0, 0].set_title('Adam Loss')
axes[0, 0].set_xlabel('Weight Decay')
axes[0, 0].set_ylabel('Learning Rate')
axes[0, 0].set_xticks(range(len(weight_decays)))
axes[0, 0].set_xticklabels(weight_decays, rotation=90)
axes[0, 0].set_yticks(range(len(learning_rates)))
axes[0, 0].set_yticklabels(learning_rates)
fig.colorbar(im, ax=axes[0, 0], orientation='vertical', fraction=0.02, pad=0.04)

im = axes[0, 1].imshow(optimizer_matrices['Adam']['accuracy'], cmap='hot', aspect='auto')
axes[0, 1].set_title('Adam Accuracy')
axes[0, 1].set_xlabel('Weight Decay')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_xticks(range(len(weight_decays)))
axes[0, 1].set_xticklabels(weight_decays, rotation=90)
axes[0, 1].set_yticks(range(len(learning_rates)))
axes[0, 1].set_yticklabels(learning_rates)
fig.colorbar(im, ax=axes[0, 1], orientation='vertical', fraction=0.02, pad=0.04)

# Plotting AdamW
im = axes[1, 0].imshow(optimizer_matrices['AdamW']['loss'], cmap='hot', aspect='auto')
axes[1, 0].set_title('AdamW Loss')
axes[1, 0].set_xlabel('Weight Decay')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_xticks(range(len(weight_decays)))
axes[1, 0].set_xticklabels(weight_decays, rotation=90)
axes[1, 0].set_yticks(range(len(learning_rates)))
axes[1, 0].set_yticklabels(learning_rates)
fig.colorbar(im, ax=axes[1, 0], orientation='vertical', fraction=0.02, pad=0.04)

im = axes[1, 1].imshow(optimizer_matrices['AdamW']['accuracy'], cmap='hot', aspect='auto')
axes[1, 1].set_title('AdamW Accuracy')
axes[1, 1].set_xlabel('Weight Decay')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_xticks(range(len(weight_decays)))
axes[1, 1].set_xticklabels(weight_decays, rotation=90)
axes[1, 1].set_yticks(range(len(learning_rates)))
axes[1, 1].set_yticklabels(learning_rates)
fig.colorbar(im, ax=axes[1, 1], orientation='vertical', fraction=0.02, pad=0.04)

plt.savefig('VGG_grid_test.png')
plt.show()
