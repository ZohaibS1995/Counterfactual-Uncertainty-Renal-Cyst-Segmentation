# general imports
import numpy as np
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm

# specific imports
from monai.losses.dice import *
from create_dataset import get_data
from create_network import get_network
from task_params import data_loader_params, patch_size
from generative.networks.nets import AutoencoderKL

def plot_slices_with_contours(array_3d, mask_3d, save_path="output.png"):
    # Determine the number of slices
    num_slices = array_3d.shape[2]

    # Create a figure for plotting
    fig, axes = plt.subplots(6, 8, figsize=(16, 12)) # Adjust this for different grid sizes
    axes = axes.flatten()

    for i in range(num_slices):
        ax = axes[i]
        slice_2d = array_3d[:, :, i]
        mask_slice = mask_3d[:, :, i]

        # Plot the slice
        ax.imshow(slice_2d, cmap='gray')

        # Plot contours for labels 1 and 2
        for label in [1, 2]:
            contour = mask_slice == label
            ax.contour(contour, colors=['red' if label == 1 else 'blue'], levels=[0.5])

        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_slices_with_contours_title(array_3d, mask_3d, save_path="output.png",sens_score = None):
    # Determine the number of slices
    num_slices = array_3d.shape[2]

    # Create a figure for plotting
    fig, axes = plt.subplots(6, 8, figsize=(16, 12)) # Adjust this for different grid sizes
    axes = axes.flatten()

    for i in range(num_slices):
        ax = axes[i]
        slice_2d = array_3d[:, :, i]
        mask_slice = mask_3d[:, :, i]

        # Plot the slice
        ax.imshow(slice_2d, cmap='gray')

        # Plot contours for labels 1 and 2
        for label in [1, 2]:
            contour = mask_slice == label
            ax.contour(contour, colors=['red' if label == 1 else 'blue'], levels=[0.5])

        ax.axis('off')

    plt.tight_layout()
    if sens_score is not None:
        plt.title("Sensitivity: " + str(sens_score))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def dice_score_v(array1, array2):
    """
    Compute the Dice score between two 3D numpy arrays.

    Parameters:
    - array1: 3D numpy array
    - array2: 3D numpy array

    Returns:
    - Dice score (float)
    """

    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")

    intersection = np.sum(array1 * array2)
    volume_sum = np.sum(array1) + np.sum(array2)

    if volume_sum == 0:
        return 1.0  # both arrays are empty, hence identical

    return 2 * intersection / volume_sum

def load_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def convert_options(config_section):
    options = type('Options', (), {})()
    for key, value in config_section.items():
        if value.lower() == 'true':
            setattr(options, key, True)
        elif value.lower() == 'false':
            setattr(options, key, False)
        elif value.isdigit():
            setattr(options, key, int(value))
        elif value.replace('.', '', 1).isdigit():
            setattr(options, key, float(value))
        else:
            setattr(options, key, value)
    return options

def print_config(config):
    for section in config.sections():
        print(f"[{section}]")
        for key in config[section]:
            print(f"{key} = {config[section][key]}")
        print()



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


def dice_score(preds, targets, epsilon=1e-6):
    preds = preds.float()
    targets = targets.float()

    intersection = torch.sum(preds * targets, dim=(0, 2, 3, 4))
    union = torch.sum(preds, dim=(0, 2, 3, 4)) + torch.sum(targets, dim=(0, 2, 3, 4))

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


def reparameterize(mu, sigma):
    epsilon = torch.randn_like(sigma)
    return mu + sigma * epsilon

def calculate_sensitivity(region_mask_gt, region_mask_pred):
    true_positive = np.sum(region_mask_gt & region_mask_pred)
    total_true = np.sum(region_mask_gt)
    return true_positive / total_true if total_true > 0 else 0

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_uncertain_model_pred(recon_data, properties, task_id):
    path_fold = r"/data/kidnAI/DynUnet/data/model_uncertain_fold2"
    lst_uncertain = [r"fold0_uncertain_0", r"fold0_uncertain_1", r"fold0_uncertain_2"]

    lst_preds = []

    for fold_u_t in lst_uncertain:
        for idx in range(390, 400):
            # initializing early stopping
            net2 = get_network(properties, task_id, path_fold,
                               f"{fold_u_t}_{idx}.pth")
            net2 = net2.to(device)
            net2.eval()

            # Generate prediction from modified reconstruction
            pred_net = net2(recon_data)
            pred_net = np.squeeze(F.softmax(pred_net, dim=1).detach().cpu().numpy())[2, ...]
            lst_preds.append(pred_net)

    pred = np.stack(lst_preds)
    pred_val = np.var(pred, axis=0)

    return pred_val

if __name__ == "__main__":


    # model save directory
    model_save_dir = r"/data/kidnAI/interpretability/check_data/models"
    check_data_test_snippets = r"/data/kidnAI/interpretability/check_data/test"

    # path to images
    store_gt_cor_pred = r"/data/kidnAI/interpretability/check_data/analysis/gt_cor_pred"
    store_gt_miss_pred = r"/data/kidnAI/interpretability/check_data/analysis/gt_miss_pred"
    store_image_path = r"/data/kidnAI/interpretability/check_data/analysis/images"
    store_analysis_sens = r"/data/kidnAI/interpretability/check_data/analysis/dice_analysis_detected_all"

    # reading configurations
    config_filename = 'config.ini'
    config = load_config(config_filename)
    options = convert_options(config['Settings'])
    options.task_id = f"{options.task_id:02}"
    args = options

    # load hyper-parameters
    task_id = args.task_id
    fold = args.fold
    val_output_dir = os.path.join(args.datalist_path, "/runs_{}_fold{}_{}/".format(task_id, fold, args.expr_name))
    log_filename = "nnunet_task{}_fold{}.log".format(task_id, fold)
    log_filename = os.path.join(val_output_dir, log_filename)
    interval = args.interval
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    multi_gpu_flag = args.multi_gpu
    amp_flag = args.amp
    lr_decay_flag = args.lr_decay
    sw_batch_size = args.sw_batch_size
    tta_val = args.tta_val
    batch_dice = args.batch_dice
    window_mode = args.window_mode
    eval_overlap = args.eval_overlap
    local_rank = args.local_rank
    determinism_flag = args.determinism_flag
    determinism_seed = args.determinism_seed

    # loading batch size
    train_batch_size = data_loader_params[task_id]["batch_size"]

    # setting the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    # loading validation dataloader
    properties, val_loader = get_data(args, mode="validation")

    # initializing early stopping
    net2 = get_network(properties, task_id, args.datalist_path,
                       "fold0_uncertain_2_399.pth")  # os.path.join(args.datalist_path, 'final_model.pth'))
    net2 = net2.to(device)
    net2.eval()

    # defining the model
    # setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Define Autoencoder KL network
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    )
    autoencoder.to(device)

    # loading the model
    model = torch.load(os.path.join(model_save_dir, "ct_checkpoint_train_cor.pt"))
    autoencoder.load_state_dict(model)
    autoencoder.eval()

    # Define sensitivity values
    sensitivity_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold = 0.05  # Tolerance for sensitivity matching

    count_c = 0
    count_miss = 0
    not_detected_val = []
    for f in tqdm(os.listdir(store_gt_miss_pred)):


        if "BOSNIAK" in f:
            temp_f = f.split("_")[0] + "_" + f.split("_")[1] + "_" + f.split("_")[2] + ".npy"
        elif "RADIOMICS" in f:
            temp_f = f.split("_")[0] + "_" + f.split("_")[1]  + ".npy"

        path_img_arr = np.load(os.path.join(store_image_path, temp_f))
        label_val = np.load(os.path.join(store_gt_miss_pred, f))

        print(f"{f}: {path_img_arr.shape}")

        if path_img_arr.shape != [128, 128, 48]:
            path_img_arr = path_img_arr[:128, :128, :48]
            label_val = label_val[:128, :128, :48]

        input_val = torch.tensor(path_img_arr[None, None, ...]).to(device)
        label_val = torch.tensor(label_val[None, None, ...]).to(device)

        # prediction
        test_pred = net2(input_val)
        sf_test_pred = np.argmax(np.squeeze(test_pred.detach().cpu().numpy()), axis=0)

        # Generator part
        reconstruction, z_mu, z_sigma = autoencoder(input_val)
        z = reparameterize(z_mu, z_sigma)

        reconstruction = autoencoder.decode(z)
        recon_test_pred = net2(reconstruction)
        recon_sf_test_pred = np.argmax(np.squeeze(recon_test_pred.detach().cpu().numpy()), axis=0)

        pred_orig = (sf_test_pred == 2) * 1
        pred_recon = (recon_sf_test_pred == 2) * 1
        orig_lab = (label_val.detach().cpu().numpy()[0, 0, ...] == 1) * 1

        print(f" Dice (pred_orig and orig_lab): {calculate_sensitivity(orig_lab, pred_orig)}")
        print(f" Dice (pred_recon and orig_lab): {calculate_sensitivity(orig_lab, pred_recon)}")
        print(f" Dice (pred_orig and pred_recon): {calculate_sensitivity(pred_orig, pred_recon)}")

        if True:
            temp_sens = calculate_sensitivity(orig_lab, pred_recon)
            save_sens = np.round(temp_sens, 2)
            fold_name = os.path.join(store_analysis_sens, f.split(".npy")[0])
            os.makedirs(fold_name, exist_ok=True)

            print(os.path.join(fold_name, f"base_{save_sens}".replace(".", "_") + ".png"))
            plot_slices_with_contours_title(reconstruction.detach().cpu().numpy()[0, 0, ...],
                                            recon_sf_test_pred,
                                            save_path=os.path.join(fold_name,
                                                                   f"base_{save_sens}".replace(".", "_") + ".png"),
                                            sens_score=save_sens,
                                            )

            np.save(os.path.join(fold_name, f"base_recon_{save_sens}".replace(".", "_") + ".npy"),
                    reconstruction.detach().cpu().numpy()[0, 0, ...])

            np.save(os.path.join(fold_name, f"base_predlab_{save_sens}".replace(".", "_") + ".npy"),
                    recon_sf_test_pred)


        detected = True
        if calculate_sensitivity(orig_lab, pred_recon) < 0.05:# and calculate_sensitivity(orig_lab, pred_orig) > 0.05:
            count_miss += 1
            detected = False
            print("*** NOT DETECTED ***")
            plot_slices_with_contours(input_val.detach().cpu().numpy()[0, 0, ...],
                                      label_val.detach().cpu().numpy()[0, 0, ...],
                                      os.path.join(check_data_test_snippets, f'gt_{count_c}.png'))

            plot_slices_with_contours(input_val.detach().cpu().numpy()[0, 0, ...],
                                      sf_test_pred,
                                      os.path.join(check_data_test_snippets, f'pred_{count_c}.png'))

            plot_slices_with_contours(reconstruction.detach().cpu().numpy()[0, 0, ...],
                                      recon_sf_test_pred,
                                      os.path.join(check_data_test_snippets, f'recon_pred_{count_c}.png'))
            print(f)
            not_detected_val.append(f)
        print(not_detected_val)
        print("\n")

        if False:
            recon_pred_tensor = (torch.argmax(recon_test_pred.squeeze(), dim=0) == 1) * 1
            cyst_pred = recon_test_pred[:, 2, :, :] * label_val[0, 0, ...]
            kid_pred = recon_test_pred[:, 1, :, :] * (torch.argmax(recon_test_pred.squeeze(), dim=0) > 0)
            kid_cyst_pred = recon_test_pred[:, 1, :, :] * label_val[0, 0, ...]

            cyst_pred_sum = torch.sum(cyst_pred)
            kid_pred_sum = torch.sum(kid_pred)
            kid_cyst_sum = torch.sum(kid_cyst_pred)

            latent_gradients = torch.autograd.grad((cyst_pred_sum), z, retain_graph=True)[0]
            latent_gradients_kid = torch.autograd.grad((kid_pred_sum), z, retain_graph=True)[0]
            latent_gradients_kid_cyst = torch.autograd.grad((kid_cyst_sum), z)[0]

            def compute_shift(alpha):
                with torch.no_grad():
                    mod_z = z + (alpha * latent_gradients) - (alpha * latent_gradients_kid_cyst) - (alpha * latent_gradients_kid)
                    mod_recon = autoencoder.decode(mod_z)
                    mod_pred = net2(mod_recon)
                    mod_seg = np.argmax(mod_pred.detach().cpu().numpy(), axis=1)[0]
                    mod_sens = calculate_sensitivity(orig_lab, (mod_seg == 2) * 1)
                return mod_sens

            # computing lower bound
            lb = -0.1
            steps = 1000
            lb_alph = np.linspace(lb, 0, steps)[::-1]
            last_pred = 1
            zero_alph_pred = compute_shift(0)
            for idx, alpha in enumerate(lb_alph):
                curr_pred = compute_shift(alpha)
                print(curr_pred)
                if curr_pred > last_pred:
                    lb = alpha
                    break
                if curr_pred < 0.03:
                    lb = alpha
                    break
                if curr_pred > zero_alph_pred:
                    lb = alpha
                    break
                last_pred = curr_pred
            print("Detected Lower Bound: ", lb)

            # computing upper bound
            rb = 0.01
            steps = 1000
            rb_alph = np.linspace(0, rb, steps)
            last_pred = 0
            zero_alph_pred = compute_shift(0)
            for idx, alpha in enumerate(rb_alph):
                curr_pred = compute_shift(alpha)
                print(curr_pred)
                if curr_pred < last_pred:
                    rb = alpha
                    break
                if curr_pred > 0.97:
                    rb = alpha
                    break
                if curr_pred < zero_alph_pred:
                    rb = alpha
                    break
                last_pred = curr_pred
            print("Detected Upper Bound: ", rb)

            if True:
                alphas = np.linspace(lb, rb, 200)[::-1]
                cont_saved = []
                for idx, alpha in enumerate(alphas):
                    modified_z = z + (alpha * latent_gradients) - (alpha * latent_gradients_kid_cyst) - (alpha * latent_gradients_kid)

                    with torch.no_grad():
                        # Reconstruct image from modified latent space
                        modified_reconstruction = autoencoder.decode(modified_z)

                        # Generate prediction from modified reconstruction
                        modified_pred = net2(modified_reconstruction)

                    # Optional: Convert to class predictions, if needed
                    modified_sf_pred = np.argmax(modified_pred.detach().cpu().numpy(), axis=1)[0]

                    temp_sens = calculate_sensitivity(orig_lab, (modified_sf_pred == 2)* 1)

                    print(temp_sens)
                    nearest_sensitivity = find_nearest(sensitivity_values, temp_sens)

                    fold_name = os.path.join(store_analysis_sens, f.split(".npy")[0])
                    os.makedirs(fold_name, exist_ok=True)

                    if (np.abs(nearest_sensitivity - temp_sens) <= threshold) and (nearest_sensitivity not in cont_saved):
                        cont_saved.append(nearest_sensitivity)
                        save_sens = np.round(temp_sens, 2)
                        plot_slices_with_contours_title(modified_reconstruction.detach().cpu().numpy()[0, 0, ...],
                                                  modified_sf_pred,
                                                  save_path=os.path.join(fold_name,
                                                                     f"{save_sens}".replace(".", "_") + ".png"),
                                                  sens_score=nearest_sensitivity,
                                                  )

                        pred_uncertain = get_uncertain_model_pred(modified_reconstruction, properties, task_id)
                        plot_slices_with_contours_title(pred_uncertain,
                                                  modified_sf_pred,
                                                  save_path=os.path.join(fold_name,
                                                                     f"uncertain_{save_sens}".replace(".", "_") + ".png"),
                                                  sens_score=nearest_sensitivity,
                                                  )

                        np.save(os.path.join(fold_name, f"recon_{save_sens}".replace(".", "_") + ".npy"),
                                modified_reconstruction.detach().cpu().numpy()[0, 0, ...])

                        np.save(os.path.join(fold_name, f"predlab_{save_sens}".replace(".", "_") + ".npy"),
                                modified_sf_pred)


            del latent_gradients
            del latent_gradients_kid_cyst
            del latent_gradients_kid

        count_c += 1

    print(not_detected_val)