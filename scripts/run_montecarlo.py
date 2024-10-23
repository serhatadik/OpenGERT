import sionna
try:
    import google.colab
    colab_compat = True
except:
    colab_compat = False

import tensorflow as tf
# Add GPU configuration at the start
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using GPU: {logical_gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU devices found. Running on CPU.")

# Rest of your imports
import datetime
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
from openge.RT.utils import (
    perturb_building_heights,
    find_highest_z_at_xy,
    perturb_building_positions,
    perturb_material_properties
)
import mitsuba as mi
import sys
import subprocess
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import gc  
from tqdm import tqdm

def check_system_cuda():
    info = {
        "Python Version": sys.version,
        "CUDA_HOME": os.environ.get("CUDA_HOME", "Not set"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "Not set"),
    }
    
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        info["nvidia-smi"] = "Available"
        info["GPU Info"] = nvidia_smi
    except:
        info["nvidia-smi"] = "Not available"
    
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"]).decode()
        info["nvcc"] = nvcc
    except:
        info["nvcc"] = "Not available"
    
    return info

system_info = check_system_cuda()
for key, value in system_info.items():
    print(f"\n{key}:")
    print(value)
# You can also try with tensorflow
print("TF GPU devices:", tf.config.list_physical_devices('GPU'))

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

devs = get_available_devices()

cuda_available = False
for dev in devs:
    if "gpu" in dev.lower():
        cuda_available = True

# Set Mitsuba variant based on CUDA availability
if cuda_available:
    try:
        mi.set_variant('cuda_ad_rgb')
        # Verify CUDA is actually working
        test_transform = mi.Transform4f()
        print("CUDA variant successfully initialized")
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        print("Falling back to LLVM variant")
        mi.set_variant('llvm_ad_rgb')
else:
    print("CUDA not available, using LLVM variant")
    mi.set_variant('llvm_ad_rgb')

print(f"Using Mitsuba variant: {mi.variant()}")

# Configure TensorFlow for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Configure TensorFlow to use the first GPU
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("TensorFlow GPU device configured successfully")
    except RuntimeError as e:
        print(f"Error configuring TensorFlow GPU: {e}")
        print("TensorFlow falling back to CPU")


# Define a list of transmitter locations
tx_locations = [
    [10, 0],
    [20, 0],
    [30, 0],
    [40, 0],
    [50, 0]
]

# Directory to save the results
output_dir = "path_gain_results"
os.makedirs(output_dir, exist_ok=True)

device = '/device:GPU:0' if physical_devices else '/device:CPU:0'
# Iterate over TX locations
for tx_loc in tx_locations:
    tx_xy = tx_loc
    # Create a subdirectory for each TX location
    tx_dir = os.path.join(output_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}")
    os.makedirs(tx_dir, exist_ok=True)
    
    # List to store path_gain tensors for this TX location
    path_gain_list = []
    
    # Initialize grid origin and cell size
    grid_origin = None
    cm_cell_size = None
    
    with tf.device(device):  # Explicitly use GPU
        scene = load_scene(sionna.rt.scene.munich)

        tx_z, tx_bldg = find_highest_z_at_xy(scene, tx_xy[0], tx_xy[1], include_ground=False)
        print(f"tx bldg: {tx_bldg}")
        basename = re.split(r'-itu', tx_bldg)[0]
        tx_z += 6

        del scene
        
    # Perform 10 perturbations
    for perturbation_index in tqdm(range(10), desc='Perturbation Progress'):
        # Load the scene
        with tf.device(device):
            scene = load_scene(sionna.rt.scene.munich)
        
            try:
                # Perturb material properties
                perturb_material_properties(scene, rel_perm_sigma=2.0, cond_sigma=0.3, verbose=False)
                
                # Perturb building heights and positions
                perturb_sigma_height = 3
                bldg_group_to_height_pert = perturb_building_heights(scene, perturb_sigma=perturb_sigma_height)
                tx_z_perturb = bldg_group_to_height_pert[basename]

                perturb_sigma_position = 1
                bldg_group_to_pos_pert = perturb_building_positions(scene, perturb_sigma=perturb_sigma_position)
                tx_xy_perturb = bldg_group_to_pos_pert[basename]
                
                tx_xy += np.array(tx_xy_perturb)
                tx_z += tx_z_perturb
                # Configure antenna array for transmitter and receiver
                scene.tx_array = PlanarArray(
                    num_rows=1,
                    num_cols=1,
                    vertical_spacing=0.7,
                    horizontal_spacing=0.5,
                    pattern="tr38901",
                    polarization="VH")
                
                scene.rx_array = PlanarArray(
                    num_rows=1,
                    num_cols=1,
                    vertical_spacing=0.5,
                    horizontal_spacing=0.5,
                    pattern="dipole",
                    polarization="cross")
                
                # Determine TX position (with height)
                #tx_z, tx_bldg = find_highest_z_at_xy(
                #    scene, tx_xy[0], tx_xy[1], include_ground=False)
                #tx_z += 6

                ### TODO: Modify the transmitter coordinates such that the bldg perturbations are
                # equally applied to the transmitter. If the bldg on which the tx is moves, the tx should
                # move equally. 
                
                # Receiver position (fixed or can be varied)
                #rx_xy = [85, 90]
                #rx_z, rx_bldg = find_highest_z_at_xy(
                #    scene, rx_xy[0], rx_xy[1], include_ground=True)
                #rx_z += 2
                
                # Create transmitter
                tx = Transmitter(
                    name="tx",
                    position=[tx_xy[0], tx_xy[1], tx_z],
                    orientation=[0, 0, 0])
                scene.add(tx)
                
                # Create receiver
                #rx = Receiver(
                #    name="rx",
                #    position=[rx_xy[0], rx_xy[1], rx_z],
                #    orientation=[0, 0, 0])
                #scene.add(rx)
                
                # Transmitter points towards receiver
                #tx.look_at(rx)
                
                # Compute coverage map
                with tf.device(device):
                    cm = scene.coverage_map(
                        max_depth=7,
                        los=True,
                        reflection=True,
                        diffraction=True,
                        cm_cell_size=(1, 1),
                        num_samples=int(1e6),
                        num_runs=1)
                
                tx_z_formatted = float(f"{tx_z:.3f}")
                # Store the path_gain tensor in the list for analysis
                path_gain_list.append(cm.path_gain.numpy())

                # Optionally, save the path_gain as a numpy array
                np_filename = f"tx_{tx_xy[0]}_{tx_xy[1]}_z_{tx_z_formatted}_" \
                            f"perturb_{perturbation_index}_path_gain.npy"
                np_filepath = os.path.join(tx_dir, np_filename)
                np.save(np_filepath, cm.path_gain.numpy())
                
                # Save grid origin and cell sizes for plotting (once)
                if grid_origin is None:
                    grid_origin = cm.center
                    cm_cell_size = cm.cell_size
                
            except Exception as e:
                print(f"An error occurred during perturbation {perturbation_index} at TX location {tx_xy}: {e}")
                continue

            finally:
                # Clean up
                if 'cm' in locals():
                    del cm
                if 'tx' in locals():
                    del tx
                if 'rx' in locals():
                    del rx
                if 'scene' in locals():
                    del scene
                tf.keras.backend.clear_session()
                gc.collect()

    # After all perturbations, process the data if any was collected
    if path_gain_list:
        # Compute mean and standard deviation across perturbations
        path_gain_array = np.stack(path_gain_list, axis=0)  # Shape: (num_perturbations, H, W)
        epsilon = 1e-20
        path_gain_array_db = 10 * np.log10(np.maximum(path_gain_array, epsilon))
        path_gain_mean_db = np.mean(path_gain_array_db, axis=0)
        path_gain_std_db = np.std(path_gain_array_db, axis=0)
        # Save mean and std path_gain
        mean_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_mean.npy")
        std_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_std.npy")
        np.save(mean_filename, path_gain_mean_db)
        np.save(std_filename, path_gain_std_db)
        
        
        # Get coverage map extent for plotting
        extent = (
            grid_origin[0],
            grid_origin[0] + cm_cell_size[0] * path_gain_mean_db.shape[1],
            grid_origin[1],
            grid_origin[1] + cm_cell_size[1] * path_gain_mean_db.shape[0]
        )
        
        # Plot mean path gain
        plt.figure(figsize=(8, 6))
        plt.imshow(
            path_gain_mean_db[0, :, :],
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label='Mean Path Gain (dB)')
        plt.title(f"Mean Path Gain for TX at {tx_xy}")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        mean_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_mean.png")
        plt.savefig(mean_fig_filename)
        plt.close()
        # Plot standard deviation of path gain
        plt.figure(figsize=(8, 6))
        plt.imshow(
            path_gain_std_db[0, :, :],
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label='Std Dev of dB Path Gain')
        plt.title(f"Std Dev of Path Gain for TX at {tx_xy}")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        std_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_std.png")
        plt.savefig(std_fig_filename)
        plt.close()
        
        # Clean up
        del path_gain_array
        del path_gain_array_db
        del path_gain_mean_db
        del path_gain_std_db
        del path_gain_list
        gc.collect()
