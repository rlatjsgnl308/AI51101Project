# +
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch VGGNet Training Test Visualization')
parser.add_argument('-dir', '--stats_dir', default="../VGG/Stats",
                    help='where is the statistic txt file? (default: ../VGG/Stats)')
args = parser.parse_args()

# Directory containing the stats files
stats_dir = args.stats_dir

# Hyperparameter settings
learning_rates = [0.0001]
weight_decays = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
optimizers = ["Adam", "AdamW"]

# Regular expression to match the stats lines
stats_pattern = re.compile(
    r'\[epoch (\d+)\] train_loss : ([\d.]+), train accu : ([\d.]+) , val_loss : ([\d.]+), val accu : ([\d.]+)'
)

# Regular expression to match the filename, allowing for scientific notation in weight decay
filename_pattern = re.compile(
    r'VGG_train_(Adam|AdamW)_(\d*\.?\d+)_([.\dEe-]+)_\d+_\d+\.txt'
)

# Initialize matrices for total_loss, recon_loss, KL_loss
num_epochs = 200  # Assuming a fixed number of epochs for visualization
train_loss_matrix = {optimizer: np.zeros((num_epochs, len(weight_decays))) for optimizer in optimizers}
train_acc_matrix = {optimizer: np.zeros((num_epochs, len(weight_decays))) for optimizer in optimizers}
test_loss_matrix = {optimizer: np.zeros((num_epochs, len(weight_decays))) for optimizer in optimizers}
test_acc_matrix = {optimizer: np.zeros((num_epochs, len(weight_decays))) for optimizer in optimizers}
count_matrix = {optimizer: np.zeros((num_epochs, len(weight_decays))) for optimizer in optimizers}

# Function to get the indices for the matrix
def get_wd_index(wd):
    for i, w in enumerate(weight_decays):
        if np.isclose(w, wd, atol=1e-10):
            return i
    return -1

# Process each file in the directory
for optimizer in optimizers:
    for filename in os.listdir(stats_dir):
        if filename.startswith(f'VGG_train_{optimizer}'):
            filepath = os.path.join(stats_dir, filename)
            match = filename_pattern.match(filename)
            if match:
                # Extract information from the filename
                extracted_optimizer = match.group(1)
                lr = float(match.group(2))
                wd = float(match.group(3))  # Ensure weight decay is parsed correctly

                print(f"Processing file: {filename}")  # Debugging print statement
                print(f"Weight decay: {wd}")  # Debugging print statement

                if wd in weight_decays:
                    wd_idx = get_wd_index(wd)

                    if wd_idx == -1:
                        print(f"Weight decay {wd} not found in predefined list.")
                        continue

                    # Read the file content
                    with open(filepath, 'r') as file:
                        lines = file.readlines()

                    # Process each epoch line
                    for line in lines:
                        stats_match = stats_pattern.match(line)
                        if stats_match:
                            epoch, train_loss, train_acc, test_loss, test_acc = map(float, stats_match.groups())
                            epoch_idx = int(epoch) - 1  # Adjust for zero-based indexing
                            
                            # Update matrices
                            train_loss_matrix[optimizer][epoch_idx, wd_idx] += train_loss
                            train_acc_matrix[optimizer][epoch_idx, wd_idx] += train_acc
                            test_loss_matrix[optimizer][epoch_idx, wd_idx] += test_loss
                            test_acc_matrix[optimizer][epoch_idx, wd_idx] += test_acc
                            count_matrix[optimizer][epoch_idx, wd_idx] += 1

# Average the results
print("Making plots...")
for optimizer in optimizers:
    train_loss_matrix[optimizer] = np.where(count_matrix[optimizer] > 0, train_loss_matrix[optimizer] / count_matrix[optimizer], np.nan)
    train_acc_matrix[optimizer] = np.where(count_matrix[optimizer] > 0, train_acc_matrix[optimizer] / count_matrix[optimizer], np.nan)
    test_loss_matrix[optimizer] = np.where(count_matrix[optimizer] > 0, test_loss_matrix[optimizer] / count_matrix[optimizer], np.nan)
    test_acc_matrix[optimizer] = np.where(count_matrix[optimizer] > 0, test_acc_matrix[optimizer] / count_matrix[optimizer], np.nan)

# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Training test statistics')

colors = {'Adam': 'red', 'AdamW': 'blue'}
transparency_levels = np.linspace(1, 0.3, len(weight_decays))

for wd_idx, wd in enumerate(weight_decays):
    for optimizer in optimizers:
        transparency = transparency_levels[wd_idx]
        axes[0,0].plot(range(1, num_epochs+1), train_loss_matrix[optimizer][:, wd_idx], label=f'{optimizer}', color=colors[optimizer], alpha=transparency)
        axes[0,1].plot(range(1, num_epochs+1), train_acc_matrix[optimizer][:, wd_idx], label=f'{optimizer}', color=colors[optimizer], alpha=transparency)
        axes[1,0].plot(range(1, num_epochs+1), test_loss_matrix[optimizer][:, wd_idx], label=f'{optimizer}', color=colors[optimizer], alpha=transparency)
        axes[1,1].plot(range(1, num_epochs+1), test_acc_matrix[optimizer][:, wd_idx], label=f'{optimizer}', color=colors[optimizer], alpha=transparency)

# Setting the titles and labels
axes[0,0].set_title('Train Loss')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')

axes[0,1].set_title('Train Accuracy')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')

axes[1,0].set_title('Test Loss')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('Loss')

axes[1,1].set_title('Test Accuracy')
axes[1,1].set_xlabel('Epoch')
axes[1,1].set_ylabel('Loss')

# Adding legend to the first plot only, as it applies to all
handles, labels = axes[0,0].get_legend_handles_labels()
unique_labels = list(dict.fromkeys(labels))  # Removing duplicates
axes[0,0].legend(handles[:len(unique_labels)], unique_labels, loc='upper right')
axes[0,1].legend(handles[:len(unique_labels)], unique_labels, loc='lower right')
axes[1,0].legend(handles[:len(unique_labels)], unique_labels, loc='upper right')
axes[1,1].legend(handles[:len(unique_labels)], unique_labels, loc='lower right')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('VGG_training_test.png')
plt.show()
