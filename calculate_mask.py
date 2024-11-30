import torch
import os
import matplotlib.pyplot as plt

def load_mask(epoch, folder='/home/4paradigm/Weather/masks'):
    """Load the mask file for a given epoch."""
    mask_path = os.path.join(folder, f'maskA_epoch{epoch}.pth')
    return torch.load(mask_path)

def flatten_tensor_list(tensor_list):
    """Flatten nested lists of tensors."""
    flat_list = []
    for item in tensor_list:
        if isinstance(item, list):
            flat_list.extend(flatten_tensor_list(item))  # Recursively flatten if nested
        else:
            flat_list.append(item)
    return flat_list

def calculate_changes(num_epochs, folder='/home/4paradigm/Weather/masks'):
    """Calculate the percentage change in masks between epochs."""
    changes_from_previous = []
    initial_mask = flatten_tensor_list(load_mask(0, folder))
    previous_mask = initial_mask  # Clone each tensor in the flattened list

    for epoch in range(1, num_epochs + 1):
        current_mask = flatten_tensor_list(load_mask(epoch, folder))

        # Calculate total elements and differences for each tensor in the list
        total_elements = sum(m.numel() for m in previous_mask)
        diff = sum((prev != curr).float().sum().item() for prev, curr in zip(previous_mask, current_mask))
        change_percentage = (diff / total_elements) * 100
        changes_from_previous.append(change_percentage)

        # Update previous mask
        previous_mask = current_mask

    return changes_from_previous

def plot_changes(num_epochs, changes_from_previous, save_path='mask_diff_changes.png'):
    """Plot the mask changes over epochs and save the plot."""
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, changes_from_previous, label="Change from Previous Epoch", marker='o', color='b')
    
    plt.xlabel("Epoch")
    plt.ylabel("Percentage Change (%)")
    plt.title("Mask Difference Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    plt.show()

# Parameters
num_epochs = 19  # Set the number of epochs to analyze
folder_path = '/home/4paradigm/Weather/masks_change/ori_85'

# Calculate changes and plot
changes_from_prev = calculate_changes(num_epochs, folder_path)
plot_changes(num_epochs, changes_from_prev)
