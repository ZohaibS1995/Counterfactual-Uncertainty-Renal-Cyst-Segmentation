import torch
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which does not require a GUI
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_slices_as_grid(array_3d, filename='slices_grid.png'):
    # Determine the number of slices
    num_slices = array_3d.shape[2]

    # Calculate the grid size
    grid_cols = math.ceil(math.sqrt(num_slices))
    grid_rows = math.ceil(num_slices / grid_cols)

    # Create a blank array to hold the grid
    slice_height, slice_width = array_3d.shape[0], array_3d.shape[1]
    grid_height = slice_height * grid_rows
    grid_width = slice_width * grid_cols
    grid_array = np.zeros((grid_height, grid_width), dtype=array_3d.dtype)

    # Copy slices into the grid array
    for i in range(num_slices):
        row = i // grid_cols
        col = i % grid_cols
        grid_array[row * slice_height:(row + 1) * slice_height, col * slice_width:(col + 1) * slice_width] = array_3d[:,
                                                                                                             :, i]

    # Normalize the grid array to the range [0, 1] if it's not already
    if np.amin(grid_array) < 0 or np.amax(grid_array) > 1:
        grid_array = (grid_array - np.amin(grid_array)) / (np.amax(grid_array) - np.amin(grid_array))

    # Use matplotlib to save the array as a PNG image
    plt.imshow(grid_array, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_slices_as_grid_with_mask(array_3d, mask_3d, filename='slices_grid.png'):
    # Determine the number of slices
    num_slices = array_3d.shape[2]

    # Calculate the grid size
    grid_cols = math.ceil(math.sqrt(num_slices))
    grid_rows = math.ceil(num_slices / grid_cols)

    # Create a blank array to hold the grid
    slice_height, slice_width = array_3d.shape[0], array_3d.shape[1]
    grid_height = slice_height * grid_rows
    grid_width = slice_width * grid_cols
    grid_array = np.zeros((grid_height, grid_width), dtype=array_3d.dtype)

    # Copy slices into the grid array
    for i in range(num_slices):
        row = i // grid_cols
        col = i % grid_cols
        grid_array[row * slice_height:(row + 1) * slice_height, col * slice_width:(col + 1) * slice_width] = array_3d[:,
                                                                                                             :, i]

    # Normalize the grid array to the range [0, 1] if it's not already
    if np.amin(grid_array) < 0 or np.amax(grid_array) > 1:
        grid_array = (grid_array - np.amin(grid_array)) / (np.amax(grid_array) - np.amin(grid_array))

    # Use matplotlib to save the array as a PNG image
    plt.imshow(grid_array, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
