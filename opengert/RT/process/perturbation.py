import os
import sys
import gc
import re
import shutil
import glob
import subprocess
import multiprocessing
from contextlib import nullcontext
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import traceback
import logging
# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm, trange

import tensorflow as tf
import json
# Check if running in Google Colab
try:
    import google.colab
    colab_compat = True
except ImportError:
    colab_compat = False

# Import Mitsuba and configure variant based on CUDA availability
import mitsuba as mi
import seaborn as sns
import pandas as pd

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
    except Exception:
        info["nvidia-smi"] = "Not available"
    
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"]).decode()
        info["nvcc"] = nvcc
    except Exception:
        info["nvcc"] = "Not available"
    
    return info

system_info = check_system_cuda()
for key, value in system_info.items():
    print(f"\n{key}:")
    print(value)

# Verify TensorFlow GPU devices
print("TF GPU devices:", tf.config.list_physical_devices('GPU'))

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

devs = get_available_devices()
cuda_available = any("gpu" in dev.lower() for dev in devs)

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

# Import Sionna and related modules
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

# Import custom utility functions and classes
from opengert.RT.utils import (
    perturb_building_heights,
    find_highest_z_at_xy,
    perturb_building_positions,
    perturb_material_properties,
    #PerturbationTracker
)

def get_missing_perturbation_indices(tx_dir: str, num_perturbations: int) -> list:
    """
    Find the indices of perturbations missing from the specified directory.
    
    Args:
        tx_dir (str): Directory to search for perturbation files
        num_perturbations (int): Total number of expected perturbations (0-indexed)
    
    Returns:
        list: Sorted list of indices missing from the directory
    """
    if not os.path.exists(tx_dir):
        return list(range(num_perturbations + 1))
        
    found_indices = set()
    pattern = re.compile(r'perturb_(\d+)_path_gain\.npy$')
    
    for filename in os.listdir(tx_dir):
        match = pattern.search(filename)
        if match:
            index = int(match.group(1))
            found_indices.add(index)
    
    # Generate the set of all expected indices
    all_indices = set(range(num_perturbations))
    
    # Find and return the missing indices
    missing_indices = sorted(list(all_indices - found_indices))
    
    return missing_indices

def load_all_perturbation_files(tx_dir: str) -> Dict[str, List[np.ndarray]]:
    """
    Load all perturbation files from the directory.
    Returns dictionary with lists of path_gain, med, ds, and K arrays.
    
    Parameters:
        tx_dir: Directory containing perturbation files
        
    Returns:
        Dictionary containing lists of numpy arrays for each metric type
    """
    path_gain_list = []
    med_list = []
    ds_list = []
    K_list = []
    
    # Get all .npy files in directory
    files = [f for f in os.listdir(tx_dir) if f.endswith('.npy')]
    
    # Sort files by perturbation index to ensure consistent ordering
    def get_perturbation_index(filename):
        match = re.search(r'perturb_(\d+)', filename)
        return int(match.group(1)) if match else -1
    
    files.sort(key=get_perturbation_index)
    
    # Categorize and load each file
    for filename in files:
        filepath = os.path.join(tx_dir, filename)
        
        try:
            # Categorize file based on name pattern
            if 'path_gain.npy' in filename:
                path_gain_list.append(np.load(filepath))
            elif 'mean_excess_delay.npy' in filename:
                med_list.append(np.load(filepath))
            elif 'delay_spread.npy' in filename:
                ds_list.append(np.load(filepath))
            elif '_K.npy' in filename:
                K_list.append(np.load(filepath))
                
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
            continue
    
    print(f"Loaded {len(path_gain_list)} path gain files")
    print(f"Loaded {len(med_list)} mean excess delay files")
    print(f"Loaded {len(ds_list)} delay spread files")
    print(f"Loaded {len(K_list)} K-factor files")
    
    return {
        'path_gain': path_gain_list,
        'med': med_list,
        'ds': ds_list,
        'K': K_list
    }

class PerturbationSimulationManager:
    def __init__(self, perturbation_config, tracker=None):
        self.perturbation_config = perturbation_config
        #self.tracker = tracker

    def process_tx_location(self, tx_xy: np.ndarray):
        """Process perturbations for a given transmitter location."""
        # Import necessary functions
        from opengert.RT.utils import (
            perturb_building_heights,
            find_highest_z_at_xy,
            perturb_building_positions,
            perturb_material_properties,
        )

        perturbation_config = self.perturbation_config
        scene_name = perturbation_config["scene_name"]
        analyze_chan_stats = perturbation_config["analyze_chan_stats"]
        batch_size = perturbation_config["batch_size"]
        output_dir = perturbation_config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        device = perturbation_config["device"]
        tx_antenna_height = perturbation_config["tx_antenna_height"]
        num_perturbations = perturbation_config["num_perturbations"]
        sim_material_perturbation = self.sim_material_perturbation = perturbation_config["sim_material_perturbation"]
        cond_sigma_ratio =  perturbation_config["cond_sigma_ratio"]
        rel_perm_sigma_ratio = perturbation_config["rel_perm_sigma_ratio"]
        sim_building_height_perturbation = self.sim_building_height_perturbation = perturbation_config["sim_building_height_perturbation"]
        perturb_sigma_height = perturbation_config["perturb_sigma_height"]
        sim_building_position_perturbation = self.sim_building_position_perturbation = perturbation_config["sim_building_position_perturbation"]
        perturb_sigma_position = perturbation_config["perturb_sigma_position"]
        verbose = perturbation_config["verbose"]
        
        # Get initial TX position
        if scene_name.lower()=="munich":
            scene_load_name = sionna.rt.scene.munich
        elif scene_name.lower()=="etoile":
            scene_load_name = sionna.rt.scene.etoile
        else:
            scene_load_name = scene_name
        self.scene_load_name = scene_load_name
        scene = load_scene(scene_load_name) 
        tx_x, tx_y, tx_z, tx_bldg = find_highest_z_at_xy(scene, tx_xy[0], tx_xy[1], include_ground=False)
        basename = re.split(r'-itu', tx_bldg)[0]
        tx_xy = [tx_x, tx_y]
        tx_z += tx_antenna_height
        print(f"tx_xy: {tx_xy}")
        print(f"tx_z: {tx_z}")
        print(f"tx building: {tx_bldg}")
        print(f"scene size: {scene.size}")
        del scene
        tx_dir = os.path.join(output_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}")
        os.makedirs(tx_dir, exist_ok=True)

        # Check for existing perturbations
        missing_indices = get_missing_perturbation_indices(tx_dir, num_perturbations)
        remaining_perturbations = len(missing_indices)
        
        if remaining_perturbations <= 0:
            print(f"All {num_perturbations} perturbations already exist in {tx_dir}")
            return
        else:
            print(f"Running {remaining_perturbations} more perturbations.")


        USE_GPU = self.perturbation_config.get('use_gpu', False)

        if USE_GPU:
            available_gpus = tf.config.list_physical_devices('GPU')
            if not available_gpus:
                raise RuntimeError("No GPUs available")

            gpu_ids = list(range(len(available_gpus)))
            num_workers = len(gpu_ids)
            print(f"Running {num_perturbations} perturbations across GPUs: {gpu_ids}")

            # Create list of tasks
            tasks = [
                (
                    pert_idx,
                    gpu_ids[pert_idx % len(gpu_ids)],
                    scene_load_name,
                    tx_xy,
                    tx_z,
                    tx_bldg,
                    basename,
                    tx_dir,
                    analyze_chan_stats,
                    batch_size,
                    sim_material_perturbation,
                    rel_perm_sigma_ratio,
                    cond_sigma_ratio,
                    sim_building_height_perturbation,
                    perturb_sigma_height,
                    sim_building_position_perturbation,
                    perturb_sigma_position,
                    verbose
                )
                for pert_idx in missing_indices
            ]
            process_func = process_perturbation_gpu
        else:
            num_workers = 4 # Use all available CPU cores
            print(f"Number of CPU cores used: {num_workers}")
            tasks = [
                (
                    pert_idx,
                    scene_load_name,
                    tx_xy,
                    tx_z,
                    tx_bldg,
                    basename,
                    tx_dir,
                    analyze_chan_stats,
                    batch_size,
                    sim_material_perturbation,
                    rel_perm_sigma_ratio,
                    cond_sigma_ratio,
                    sim_building_height_perturbation,
                    perturb_sigma_height,
                    sim_building_position_perturbation,
                    perturb_sigma_position,
                    verbose
                )
                for pert_idx in missing_indices
            ]
            process_func = process_perturbation_cpu

        if not tasks:
            print("No new perturbations to process")
            return
            
        # Run perturbations
        results, path_gain_list, med_list, ds_list, K_list, grid_origin, cm_cell_size, grid_origin2, cm_cell_size2 = \
            self.run_perturbations(num_workers, tasks, process_func, tx_xy)
        perturbation_config["cm_cell_size"] = cm_cell_size
        perturbation_config["cm_cell_size2"] = cm_cell_size2
        perturbation_config["grid_origin"] = grid_origin.numpy().tolist()
        perturbation_config["grid_origin2"] = grid_origin2.numpy().tolist()
        perturbation_config["tx_loc"] = [*tx_xy, tx_z]
        perturbation_config["tx_dir"] = tx_dir
        with open(os.path.join(tx_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(perturbation_config, f, indent=4)
        
        all_results = load_all_perturbation_files(tx_dir)

        if all_results['path_gain']:
            # Process all path gain results
            self.process_path_gain_results(all_results['path_gain'], grid_origin, cm_cell_size, tx_xy, tx_dir)

        if all_results['med'] and all_results['ds'] and all_results['K']:
            # Process all channel statistics results
            self.process_channel_stats_results(all_results['med'], 'med', grid_origin2, cm_cell_size2, tx_xy, tx_dir)
            self.process_channel_stats_results(all_results['ds'], 'ds', grid_origin2, cm_cell_size2, tx_xy, tx_dir)
            self.process_channel_stats_results(all_results['K'], 'K', grid_origin2, cm_cell_size2, tx_xy, tx_dir)

        drjit_cache = os.path.expanduser("~/.drjit")
        if os.path.exists(drjit_cache):
            for root, dirs, files in os.walk(drjit_cache, topdown=False):
                # Remove files that don't start with .nfs
                for name in files:
                    if not name.startswith('.nfs'):
                        file_path = os.path.join(root, name)
                        os.remove(file_path)

    def run_perturbations(self, num_workers, tasks, process_func, tx_xy):
        """Run perturbations using multiprocessing."""
        results = []
        path_gain_list = []
        med_list = []
        ds_list = []
        K_list = []
        grid_origin = None
        cm_cell_size = None
        grid_origin2 = None
        cm_cell_size2 = None

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_func, task_args) for task_args in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing TX at {tx_xy}'):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"A task raised an exception: {exc}")
                    traceback.print_exc()
                    continue
                #result = future.result()
                results.append(result)
                
                if result['success']:
                    path_gain_list.append(result['path_gain'])
                    if grid_origin is None:
                        grid_origin = result['grid_origin']['path_gain']
                        cm_cell_size = result['cm_cell_size']['path_gain']
                        grid_origin2 = result['grid_origin']['channel_stats']
                        cm_cell_size2 = result['cm_cell_size']['channel_stats']
                    perturbation_number = result['perturbation_data']['perturbation_number']
                    if self.perturbation_config["analyze_chan_stats"]:
                        med_list.append(result['channel_stats']['mean_excess_delays'])
                        ds_list.append(result['channel_stats']['delay_spreads'])
                        K_list.append(result['channel_stats']['Ks'])
                    #self.tracker.update(**result['perturbation_data'])
                    if self.perturbation_config["verbose"]:
                        print(f"Completed perturbation {perturbation_number}")
                else:
                    print(f"Perturbation {result['perturbation_data']['perturbation_number']} failed.")

        # Sort results by perturbation index to maintain order
        results.sort(key=lambda x: x['perturbation_data']['perturbation_number'])

        return results, path_gain_list, med_list, ds_list, K_list, grid_origin, cm_cell_size, grid_origin2, cm_cell_size2

    def process_path_gain_results(
            self, path_gain_list, grid_origin, cm_cell_size, tx_xy, tx_dir
        ):
        """Process and plot path gain results."""
        # Stack the path gain arrays
        stacked_path_gain = np.stack(path_gain_list, axis=0)
        
        # Create a mask for broken links (values equal to 0.0)
        epsilon = 0.0
        broken_links_mask = (stacked_path_gain == epsilon)
        
        # Count frequency of broken links at each position
        broken_links_frequency = np.sum(broken_links_mask, axis=0)
        
        # Create mask for locations with fewer than maximum broken links
        max_broken_links = np.max(broken_links_frequency)
        valid_locations_mask = (broken_links_frequency < max_broken_links)
        
        # Replace broken links with nan for statistics calculation
        path_gain_masked = stacked_path_gain.copy()
        path_gain_masked[broken_links_mask] = np.nan
        
        # Convert to dB, ignoring nan values
        path_gain_array_db = 10 * np.log10(path_gain_masked)
        
        # Calculate statistics (nan values will be automatically excluded)
        if path_gain_array_db.size == 0 or np.all(np.isnan(path_gain_array_db)):
            # Return default values or raise exception based on your needs
            path_gain_mean_db = np.nan  # or some other default value
            path_gain_var_db = np.nan   # or some other default value
        else:
            path_gain_mean_db = np.nanmean(path_gain_array_db, axis=0)
            path_gain_std_db = np.nanstd(path_gain_array_db, axis=0)
            
        # Apply valid locations mask to statistics arrays
        path_gain_mean_db_masked = np.where(valid_locations_mask, path_gain_mean_db, np.nan)
        path_gain_std_db_masked = np.where(valid_locations_mask, path_gain_std_db, np.nan)
        broken_links_frequency_masked = np.where(valid_locations_mask, broken_links_frequency, np.nan)

        tx_x_formatted = float(f"{tx_xy[0]:.1f}")
        tx_y_formatted = float(f"{tx_xy[1]:.1f}")
        # Save results including broken links frequency
        mean_filename = os.path.join(tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_path_gain_mean.npy")
        std_filename = os.path.join(tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_path_gain_std.npy")
        freq_filename = os.path.join(tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_broken_links_freq.npy")
        np.save(mean_filename, path_gain_mean_db_masked)
        np.save(std_filename, path_gain_std_db_masked)
        np.save(freq_filename, broken_links_frequency_masked)
        
        # Get coverage map extent for plotting
        extent = (
            grid_origin[0] - cm_cell_size[0] * path_gain_mean_db.shape[1] / 2,
            grid_origin[0] + cm_cell_size[0] * path_gain_mean_db.shape[1] / 2,
            grid_origin[1] - cm_cell_size[1] * path_gain_mean_db.shape[0] / 2,
            grid_origin[1] + cm_cell_size[1] * path_gain_mean_db.shape[0] / 2
        )
        
        # Plot mean path gain
        plt.figure(figsize=(8, 6))
        plt.imshow(
            path_gain_mean_db_masked,
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label='Mean Path Gain (dB)')
        plt.plot(tx_xy[0], tx_xy[1], 'b*', markersize=10, label='TX') # TX
        plt.title(f"Mean Path Gain for TX at {[tx_x_formatted, tx_y_formatted]}\nPerturb Material:{self.sim_material_perturbation}, Height: {self.sim_building_height_perturbation}, Location: {self.sim_building_position_perturbation}")
        plt.xlabel('Local X Coordinates [m]')
        plt.ylabel('Local Y Coordinates [m]')
        mean_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_path_gain_mean.png")
        plt.savefig(mean_fig_filename)
        plt.close()
        self.summarize_array_statistics(path_gain_mean_db_masked, tx_xy, extent, filename=mean_fig_filename[:-4] + "_stats" + ".png", arr_name="Path Gain Mean")
        
        # Plot standard deviation of path gain
        plt.figure(figsize=(8, 6))
        plt.imshow(
            path_gain_std_db_masked,
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label='Std Dev of dB Path Gain')
        plt.plot(tx_xy[0], tx_xy[1], 'b*', markersize=10, label='TX') # TX
        plt.title(f"Std Dev of Path Gain for TX at {[tx_x_formatted, tx_y_formatted]}\nPerturb Material:{self.sim_material_perturbation}, Height: {self.sim_building_height_perturbation}, Location: {self.sim_building_position_perturbation}")
        plt.xlabel('Local X Coordinates [m]')
        plt.ylabel('Local Y Coordinates [m]')
        std_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_path_gain_std.png")
        plt.savefig(std_fig_filename)
        plt.close()
        self.summarize_array_statistics(path_gain_std_db_masked, tx_xy, extent, filename=std_fig_filename[:-4] + "_stats" + ".png", arr_name="Path Gain Standard Deviation")
        
        # Plot broken links frequency
        plt.figure(figsize=(8, 6))
        plt.imshow(
            broken_links_frequency_masked,
            extent=extent,
            origin='lower',
            cmap='YlOrRd',  # Different colormap to distinguish from other plots
            aspect='auto')
        plt.colorbar(label='Number of Broken Links')
        plt.plot(tx_xy[0], tx_xy[1], 'b*', markersize=10, label='TX') # TX
        plt.title(f"Broken Link Count for TX at {[tx_x_formatted, tx_y_formatted]}\nPerturb Material:{self.sim_material_perturbation}, Height: {self.sim_building_height_perturbation}, Location: {self.sim_building_position_perturbation}")
        plt.xlabel('Local X Coordinates [m]')
        plt.ylabel('Local Y Coordinates [m]')
        freq_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_broken_links_freq.png")
        plt.savefig(freq_fig_filename)
        plt.close()
        self.summarize_array_statistics(broken_links_frequency_masked, tx_xy, extent, filename=freq_fig_filename[:-4] + "_stats" + ".png", arr_name = "Broken Link Count")
        
        # Clean up
        del stacked_path_gain
        del path_gain_masked
        del path_gain_array_db
        del path_gain_mean_db
        del path_gain_std_db
        del broken_links_mask
        del broken_links_frequency
        del path_gain_list
        gc.collect()

    def process_channel_stats_results(
        self, stat_list, stat_name, grid_origin, cm_cell_size, tx_xy, tx_dir
        ):
        """Process and plot channel statistics results."""
        # Compute statistics and create plots
        stat_stck = np.stack(stat_list, axis=0)
        
        # Create masks for invalid data
        all_nan_mask = np.all(np.isnan(stat_stck), axis=0)
        all_zero_mask = np.all(stat_stck == 0.0, axis=0)
        valid_locations_mask = ~(all_nan_mask | all_zero_mask)  # Combine both masks
        
        # Compute statistics
        stat_mean = np.mean(stat_stck, axis=0)
        stat_std = np.std(stat_stck, axis=0)
        
        # Apply mask to statistics arrays
        stat_mean_masked = np.where(valid_locations_mask, stat_mean, np.nan)
        stat_std_masked = np.where(valid_locations_mask, stat_std, np.nan)

        # Save results
        tx_x_formatted = float(f"{tx_xy[0]:.1f}")
        tx_y_formatted = float(f"{tx_xy[1]:.1f}")
        mean_filename = os.path.join(tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_{stat_name}_mean.npy")
        std_filename = os.path.join(tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_{stat_name}_std.npy")
        np.save(mean_filename, stat_mean_masked)
        np.save(std_filename, stat_std_masked)
        
        # Get coverage map extent for plotting
        extent = (
            grid_origin[0] - cm_cell_size[0] * stat_mean.shape[1] / 2,
            grid_origin[0] + cm_cell_size[0] * stat_mean.shape[1] / 2,
            grid_origin[1] - cm_cell_size[1] * stat_mean.shape[0] / 2,
            grid_origin[1] + cm_cell_size[1] * stat_mean.shape[0] / 2
        )
        if stat_name.lower() == "ds":
            arr_name = "Delay Spread"
        elif stat_name.lower() == "med":
            arr_name = "Mean Excess Delay"
        elif stat_name == "K":
            arr_name = "Rician Factor"

        # Plot mean
        plt.figure(figsize=(8, 6))
        if stat_name == "K":
            # For K factor, use logarithmic scale
            # Handle zero, negative, and NaN values
            data_for_plot = stat_mean_masked.copy()
            valid_data = data_for_plot[~np.isnan(data_for_plot) & (data_for_plot > 0)]
            
            if len(valid_data) > 0:
                min_positive = np.min(valid_data)
                epsilon = min_positive / 10
            else:
                epsilon = 1e-10
                
            # Replace NaN, zero, and negative values with epsilon
            data_for_plot[np.isnan(data_for_plot) | (data_for_plot <= 0)] = epsilon
            
            norm = LogNorm(vmin=epsilon, vmax=np.max(data_for_plot))
            im = plt.imshow(
                data_for_plot,
                extent=extent,
                origin='lower',
                cmap='hot',
                aspect='auto',
                norm=norm)
        else:
            # For other metrics, use linear scale
            im = plt.imshow(
                stat_mean_masked,
                extent=extent,
                origin='lower',
                cmap='hot',
                aspect='auto')

        plt.colorbar(im, label=f'Mean {stat_name.upper()}')
        plt.plot(tx_xy[0], tx_xy[1], 'b*', markersize=10, label='TX') # TX
        plt.title(f"Mean {stat_name.upper()} for TX at {[tx_x_formatted, tx_y_formatted]}\n(Excluding All-NaN and All-Zero Locations)\nPerturb Material:{self.sim_material_perturbation}, Height: {self.sim_building_height_perturbation}, Location: {self.sim_building_position_perturbation}")
        plt.xlabel('Local X Coordinates [m]')
        plt.ylabel('Local Y Coordinates [m]')
        mean_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_{stat_name}_mean.png")
        plt.savefig(mean_fig_filename)
        plt.close()

        self.summarize_array_statistics(stat_mean_masked, tx_xy, extent, filename=mean_fig_filename[:-4] + "_stats" + ".png", arr_name = arr_name + " Mean", log_scale_y=True)

        # Plot standard deviation
        plt.figure(figsize=(8, 6))
        if stat_name == "K":
            # For K factor, use logarithmic scale
            # Handle zero, negative, and NaN values
            data_for_plot = stat_std_masked.copy()
            valid_data = data_for_plot[~np.isnan(data_for_plot) & (data_for_plot > 0)]
            
            if len(valid_data) > 0:
                min_positive = np.min(valid_data)
                epsilon = min_positive / 10
            else:
                epsilon = 1e-10
                
            # Replace NaN, zero, and negative values with epsilon
            data_for_plot[np.isnan(data_for_plot) | (data_for_plot <= 0)] = epsilon
            
            norm = LogNorm(vmin=epsilon, vmax=np.max(data_for_plot))
            im = plt.imshow(
                data_for_plot,
                extent=extent,
                origin='lower',
                cmap='hot',
                aspect='auto',
                norm=norm)
        else:
            # For other metrics, use linear scale
            im = plt.imshow(
                stat_std_masked,
                extent=extent,
                origin='lower',
                cmap='hot',
                aspect='auto')

        plt.colorbar(im, label=f'Std Dev of {stat_name.upper()}')
        plt.plot(tx_xy[0], tx_xy[1], 'b*', markersize=10, label='TX') # TX
        plt.title(f"Std Dev of {stat_name.upper()} for TX at {[tx_x_formatted, tx_y_formatted]}\n(Excluding All-NaN and All-Zero Locations)\nPerturb Material:{self.sim_material_perturbation}, Height: {self.sim_building_height_perturbation}, Location: {self.sim_building_position_perturbation}")
        plt.xlabel('Local X Coordinates [m]')
        plt.ylabel('Local Y Coordinates [m]')
        std_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_x_formatted}_{tx_y_formatted}_{stat_name}_std.png")
        plt.savefig(std_fig_filename)
        plt.close()
        self.summarize_array_statistics(stat_std_masked, tx_xy, extent, filename=std_fig_filename[:-4] + "_stats" + ".png", arr_name = arr_name + " Standard Deviation", log_scale_y=True)
        # Clean up
        del stat_stck
        del stat_mean
        del stat_std
        del stat_list
        gc.collect()

    def summarize_array_statistics(self, arr, tx_coordinate, rx_extent, filename="array_statistics.png", arr_name="", log_scale_y=False):

        if arr.size == 0 or np.all(np.isnan(arr)):
            print(f"Warning: Array is empty or contains all NaN values for {arr_name}")
            return None

        # Ensure NaNs are ignored in statistics calculations
        mean_val = np.nanmean(arr)
        median_val = np.nanmedian(arr)
        std_dev = np.nanstd(arr)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)

        # Create the receiver coordinates based on rx_extent
        x = np.linspace(rx_extent[0], rx_extent[1], arr.shape[1])
        y = np.linspace(rx_extent[2], rx_extent[3], arr.shape[0])
        rx_x, rx_y = np.meshgrid(x, y)

        # Calculate distances for each point in the grid
        distances = np.sqrt((rx_x - tx_coordinate[0])**2 + (rx_y - tx_coordinate[1])**2)

        # Flatten the arrays and remove NaN values from arr and distances
        arr_flat = arr.flatten()
        distances_flat = distances.flatten()
        valid_mask = ~np.isnan(arr_flat)
        arr_flat = arr_flat[valid_mask]
        distances_flat = distances_flat[valid_mask]

        if len(arr_flat) == 0:
            print(f"Warning: No valid data points after filtering for {arr_name}")
            return None

        if log_scale_y:
            nonzero_mask = arr_flat > 0
            if not np.any(nonzero_mask):
                print(f"Warning: No positive values found for log scale in {arr_name}")
                return None
            min_nonzero_val = np.min(arr_flat[nonzero_mask])
            arr_flat = np.where(arr_flat <= 0, min_nonzero_val * 0.1, arr_flat)

        try:
            tx_x_formatted = float(f"{tx_coordinate[0]:.1f}")
            tx_y_formatted = float(f"{tx_coordinate[1]:.1f}")
            scene_name = self.perturbation_config["scene_name"]
            # Set up the figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{arr_name}, Scene: {scene_name}, TX at {[tx_x_formatted, tx_y_formatted]}")

            # Scatter plot of values as a function of distance
            axes[0].scatter(distances_flat, arr_flat, alpha=0.4, edgecolors="black")
            axes[0].set_xlabel("Distance from TX")
            axes[0].set_ylabel(arr_name)

            # Apply logarithmic scale if specified
            if log_scale_y:
                axes[0].set_yscale("log")
                axes[1].set_yscale("log")

            # Histogram of the array values
            axes[1].hist(arr_flat, bins=30, color="skyblue", edgecolor="black")
            axes[1].set_xlabel(f"{arr_name}")
            axes[1].set_ylabel("Count")

            # Save the figure to file
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(filename)
            plt.close(fig)

            # Display summary statistics
            stats_df = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Standard Deviation", "Min", "Max"],
                "Value": [mean_val, median_val, std_dev, min_val, max_val]
            })
            print(f"\nSummary Statistics for {arr_name}:")
            print(stats_df.to_string(index=False))

            return mean_val, median_val, std_dev, min_val, max_val

        except Exception as e:
            print(f"Error generating plots and statistics for {arr_name}: {str(e)}")
            plt.close(fig) 
            return None

def process_perturbation_gpu(args) -> Dict[str, Any]:
    """Process a single perturbation on a specific GPU."""
    # Extract arguments
    (perturbation_index, gpu_id, scene_load_name, tx_xy, tx_z, tx_bldg, basename,
     tx_dir, analyze_chan_stats, batch_size, sim_material_perturbation, rel_perm_sigma_ratio, cond_sigma_ratio, sim_building_height_perturbation, perturb_sigma_height, sim_building_position_perturbation, perturb_sigma_position,
     verbose) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = '/device:GPU:0'

    if verbose:
        print(f"Processing perturbation {perturbation_index} on GPU {gpu_id}")

    # Create an instance of PerturbationProcessor
    processor = PerturbationProcessor(
        perturbation_index=perturbation_index,
        scene_load_name=scene_load_name,
        tx_xy=tx_xy,
        tx_z=tx_z,
        tx_bldg=tx_bldg,
        basename=basename,
        tx_dir=tx_dir,
        analyze_chan_stats=analyze_chan_stats,
        batch_size=batch_size,
        sim_material_perturbation=sim_material_perturbation,
        rel_perm_sigma_ratio=rel_perm_sigma_ratio,
        cond_sigma_ratio=cond_sigma_ratio,
        sim_building_height_perturbation=sim_building_height_perturbation,
        perturb_sigma_height=perturb_sigma_height,
        sim_building_position_perturbation=sim_building_position_perturbation,
        perturb_sigma_position=perturb_sigma_position,
        device=device,
        verbose=verbose
    )
    return processor.process_perturbation_core()

def process_perturbation_cpu(args) -> Dict[str, Any]:
    """Process a single perturbation for CPU parallel processing."""
    # Extract arguments
    (perturbation_index, scene_load_name, tx_xy, tx_z, tx_bldg, basename,
     tx_dir, analyze_chan_stats, batch_size, sim_material_perturbation, rel_perm_sigma_ratio, cond_sigma_ratio, sim_building_height_perturbation, perturb_sigma_height, sim_building_position_perturbation, perturb_sigma_position,
     verbose) = args

    if verbose:
        print(f"Starting perturbation {perturbation_index}")

    processor = PerturbationProcessor(
        perturbation_index=perturbation_index,
        scene_load_name=scene_load_name,
        tx_xy=tx_xy,
        tx_z=tx_z,
        tx_bldg=tx_bldg,
        basename=basename,
        tx_dir=tx_dir,
        analyze_chan_stats=analyze_chan_stats,
        batch_size=batch_size,
        sim_material_perturbation=sim_material_perturbation,
        rel_perm_sigma_ratio=rel_perm_sigma_ratio,
        cond_sigma_ratio=cond_sigma_ratio,
        sim_building_height_perturbation=sim_building_height_perturbation,
        perturb_sigma_height=perturb_sigma_height,
        sim_building_position_perturbation=sim_building_position_perturbation,
        perturb_sigma_position=perturb_sigma_position,
        device=None,  # No device specification for CPU
        verbose=verbose
    )

    result = processor.process_perturbation_core()

    if verbose:
        print(f"Completed perturbation {perturbation_index}")

    return result


class PerturbationProcessor:
    def __init__(
        self,
        perturbation_index: int,
        scene_load_name: str,
        tx_xy: np.ndarray,
        tx_z: float,
        tx_bldg: str,
        basename: str,
        tx_dir: str,
        analyze_chan_stats: bool,
        batch_size: int,
        sim_material_perturbation: bool,
        rel_perm_sigma_ratio: Optional[float],
        cond_sigma_ratio: Optional[float],
        sim_building_height_perturbation: bool,
        perturb_sigma_height: Optional[float],
        sim_building_position_perturbation: bool,
        perturb_sigma_position: Optional[float],
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.perturbation_index = perturbation_index
        self.scene_load_name = scene_load_name
        self.tx_xy = tx_xy
        self.tx_z = tx_z
        self.tx_bldg = tx_bldg
        self.basename = basename
        self.tx_dir = tx_dir
        self.analyze_chan_stats = analyze_chan_stats
        self.batch_size = batch_size
        self.sim_material_perturbation = sim_material_perturbation
        self.rel_perm_sigma_ratio = rel_perm_sigma_ratio
        self.cond_sigma_ratio = cond_sigma_ratio
        self.sim_building_height_perturbation = sim_building_height_perturbation
        self.perturb_sigma_height = perturb_sigma_height
        self.sim_building_position_perturbation = sim_building_position_perturbation
        self.perturb_sigma_position = perturb_sigma_position
        self.device = device
        self.verbose = verbose

    def process_perturbation_core(self) -> Dict[str, Any]:
        """Core processing function shared between GPU and CPU implementations with retry mechanism."""
        max_retries = 3  
        attempt = 0
        while True:
            attempt += 1
            context = tf.device(self.device) if self.device else nullcontext()
            
            # Dictionary to track created variables for cleanup
            created_vars = {
                'scene': None,
                'pert_tx_xy': None,
                'pert_tx_z': None,
                'perturbation_data': None,
                'result': None,
                'result_dict': None
            }
            
            try:
                with context:
                    # Load the scene
                    created_vars['scene'] = load_scene(self.scene_load_name)
                    
                    # Apply perturbations
                    created_vars['pert_tx_xy'], created_vars['pert_tx_z'], created_vars['perturbation_data'] = \
                        self.apply_perturbations(created_vars['scene'])
                    
                    # Configure the scene
                    created_vars['scene'] = self.configure_scene(
                        created_vars['scene'], 
                        created_vars['pert_tx_xy'], 
                        created_vars['pert_tx_z']
                    )
                    
                    # Process coverage maps and channel statistics
                    created_vars['result'] = self.process_coverage_and_stats(
                        created_vars['scene'],
                        created_vars['pert_tx_xy'],
                        created_vars['pert_tx_z']
                    )
                    
                    # Combine results
                    created_vars['result_dict'] = {
                        'success': True,
                        'perturbation_data': {
                            'perturbation_number': self.perturbation_index,
                            'tx_position': [
                                created_vars['pert_tx_xy'][0],
                                created_vars['pert_tx_xy'][1],
                                created_vars['pert_tx_z']
                            ],
                            'tx_building': self.tx_bldg,
                            **created_vars['perturbation_data']
                        },
                        **created_vars['result']
                    }
                    
                    if attempt > 1:
                        logging.info(f"Successfully completed processing after {attempt} attempts")
                    
                    return created_vars['result_dict']

            except Exception as e:
                logging.error(f"Attempt {attempt} failed for task {self.perturbation_index}: {e}")
                traceback_str = traceback.format_exc()
                logging.error(traceback_str)
                
                # Clean up created variables
                for var_name, var in created_vars.items():
                    if var is not None:
                        try:
                            if hasattr(var, 'cleanup'):
                                var.cleanup()
                            del var
                            logging.debug(f"Cleaned up variable: {var_name}")
                        except Exception as cleanup_error:
                            logging.warning(f"Error cleaning up {var_name}: {cleanup_error}")
                
                # Force garbage collection after cleanup
                gc.collect()
                
                if attempt >= max_retries:
                    logging.error(f"Max retries ({max_retries}) reached. Raising final exception.")
                    raise

    def apply_perturbations(self, scene) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Apply various perturbations to the scene."""

        bldg_mat = {}
        if self.sim_material_perturbation:
            bldg_mat = perturb_material_properties(scene, material_types_known=True, verbose=False, rel_perm_sigma_ratio=self.rel_perm_sigma_ratio, cond_sigma_ratio=self.cond_sigma_ratio)

        # Perturb building heights
        bldg_group_to_height_pert = {}
        pert_tx_z = float(self.tx_z)
        if self.sim_building_height_perturbation:
            perturb_sigma_height = self.perturb_sigma_height
            bldg_group_to_height_pert = perturb_building_heights(scene, perturb_sigma=perturb_sigma_height)
            tx_z_perturb = bldg_group_to_height_pert[self.basename]
            pert_tx_z += tx_z_perturb

        # Perturb building positions
        bldg_group_to_pos_pert = {}
        pert_tx_xy = np.array(self.tx_xy, dtype=np.float64)
        if self.sim_building_position_perturbation:
            perturb_sigma_position = self.perturb_sigma_position
            bldg_group_to_pos_pert = perturb_building_positions(scene, perturb_sigma=perturb_sigma_position)
            tx_xy_perturb = bldg_group_to_pos_pert[self.basename]
            pert_tx_xy += np.array(tx_xy_perturb)

        perturbation_data = {
            'bldg_mat': bldg_mat if self.sim_material_perturbation else None,
            'bldg_group_to_height_pert': bldg_group_to_height_pert if self.sim_building_height_perturbation else None,
            'bldg_group_to_pos_pert': bldg_group_to_pos_pert if self.sim_building_position_perturbation else None
        }

        return pert_tx_xy, pert_tx_z, perturbation_data

    def configure_scene(self, scene, pert_tx_xy: np.ndarray, pert_tx_z: float):
        """Configure antenna arrays and transmitter in the scene."""
        scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.7,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )
        
        scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V"
        )
        
        tx = Transmitter(
            name="tx",
            position=[pert_tx_xy[0], pert_tx_xy[1], pert_tx_z],
            orientation=[0, 0, 0]
        )
        scene.add(tx)
        
        len_ris = len(scene._ris)
        if len_ris:
            for ris_obj in scene._ris:
                scene.remove(ris_obj)

        return scene

    def process_coverage_and_stats(
        self, scene, pert_tx_xy: np.ndarray, pert_tx_z: float
    ) -> Dict[str, Any]:
        """Process coverage maps and channel statistics."""
        context = tf.device(self.device) if self.device else nullcontext()
        cm_cell_size = (5, 5)
        with context:
            # Initial coverage map
            cm = scene.coverage_map(
                max_depth=5,
                los=True,
                reflection=True,
                diffraction=True,
                ris=False,
                cm_cell_size=cm_cell_size,
                num_samples=int(1e6),
                check_scene=False,
                num_runs=1
            )
            
            cm_center = cm.center
            path_gain = cm.path_gain.numpy()[0, :, :].copy()

            del cm
            gc.collect()
            
            # Save path gain
            self.tx_x_formatted = tx_x_formatted = float(f"{pert_tx_xy[0]:.1f}")
            self.tx_y_formatted = tx_y_formatted = float(f"{pert_tx_xy[1]:.1f}")
            self.tx_z_formatted = tx_z_formatted = float(f"{pert_tx_z:.3f}")
            np_filename = f"tx_x_{tx_x_formatted}_y_{tx_y_formatted}_z_{tx_z_formatted}_perturb_{self.perturbation_index}_path_gain.npy"
            np_filepath = os.path.join(self.tx_dir, np_filename)
            np.save(np_filepath, path_gain)
            
            channel_stats = None
            cm2_center = None
            cm2_cell_size = None
            
            if self.analyze_chan_stats:
                channel_stats, cm2_center, cm2_cell_size = self.process_channel_statistics(scene)

            result_dict = {
            'path_gain': path_gain,
            'grid_origin': {
                'path_gain': cm_center,
                'channel_stats': cm2_center
            },
            'cm_cell_size': {
                'path_gain': cm_cell_size,
                'channel_stats': cm2_cell_size
            },
            'channel_stats': channel_stats
            }
            del path_gain
            if channel_stats:
                del channel_stats
            gc.collect()
        return result_dict

    def process_channel_statistics(
        self, scene
    ) -> Tuple[Dict[str, np.ndarray], Any, Tuple[int, int]]:
        """Process detailed channel statistics."""
        cm2_cell_size = (20, 20)
        context = tf.device(self.device) if self.device else nullcontext()

        with context:
            # Dummy coverage map to find out the receiver locations
            cm2 = scene.coverage_map(
                max_depth=5,
                los=True,
                reflection=True,
                diffraction=True,
                ris=False,
                cm_cell_size=cm2_cell_size,
                num_samples=int(1),
                num_runs=1
            )
            
        cell_centers = cm2.cell_centers.numpy()
        cell_centers_flat = cell_centers.reshape(-1, 3)
        cell_centers_shape = cell_centers.shape[:2]
        
        delay_spreads, mean_excess_delays, Ks = self.process_batches(
            scene, self.batch_size, cell_centers_flat, cell_centers_shape, cm2.orientation
        )

        np_filename = f"tx_x_{self.tx_x_formatted}_y_{self.tx_y_formatted}_z_{self.tx_z_formatted}_perturb_{self.perturbation_index}_delay_spread.npy"
        np_filepath = os.path.join(self.tx_dir, np_filename)
        np.save(np_filepath, delay_spreads)
        np_filename = f"tx_x_{self.tx_x_formatted}_y_{self.tx_y_formatted}_z_{self.tx_z_formatted}_perturb_{self.perturbation_index}_mean_excess_delay.npy"
        np_filepath = os.path.join(self.tx_dir, np_filename)
        np.save(np_filepath, mean_excess_delays)
        np_filename = f"tx_x_{self.tx_x_formatted}_y_{self.tx_y_formatted}_z_{self.tx_z_formatted}_perturb_{self.perturbation_index}_K.npy"
        np_filepath = os.path.join(self.tx_dir, np_filename)
        np.save(np_filepath, Ks)

        cm2_center = cm2.center
        del cm2
        gc.collect()

        return (
            {
                'delay_spreads': delay_spreads,
                'mean_excess_delays': mean_excess_delays,
                'Ks': Ks
            },
            cm2_center,
            cm2_cell_size
        )

    def process_batches(
        self, scene, batch_size, cell_centers_flat: np.ndarray, cell_centers_shape: Tuple[int, int], orientation: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process batches of receivers for channel statistics."""
        delay_spreads = []
        mean_excess_delays = []
        Ks = []
        num_cells = len(cell_centers_flat)
        #num_cells = 100
        total_batches = int(np.ceil(num_cells / batch_size))
        
        for i in range(0, num_cells, batch_size):
            current_batch = i // batch_size + 1
            print(f"Processing batch {current_batch} of {total_batches}")
            
            batch_centers = cell_centers_flat[i:i+batch_size]
            
            # Add receivers
            for idx, cell_center in enumerate(batch_centers):
                rx = Receiver(
                    name=f"rx_{idx}",
                    position=cell_center,
                    orientation=orientation
                )
                scene.add(rx)

            # Process batch
            delay_spread, mean_delay, K = self.process_single_batch(scene)

            # Append results
            delay_spreads.extend(delay_spread)
            mean_excess_delays.extend(mean_delay)
            Ks.extend(K)
            
            # Cleanup
            for idx in range(len(batch_centers)):
                scene.remove(f"rx_{idx}")
            gc.collect()
        
        # Reshape results
        return self.reshape_results(
            delay_spreads,
            mean_excess_delays,
            Ks,
            cell_centers_shape,
            num_cells
        )

    def process_single_batch(self, scene) -> Tuple[List[float], List[float]]:
        """Process a single batch of receivers."""
        paths = scene.compute_paths(
            max_depth=5,
            num_samples=int(1e6),
            los=True,
            reflection=True,
            diffraction=True,
            scattering=False,
            ris=False,
            edge_diffraction=True,
            check_scene=False
        )
        paths.normalize_delays = False
        a, tau = paths.cir(num_paths=50)
        #print(paths.types)
        del paths
        gc.collect()
        a = a[0, :, 0, 0, 0, :, 0]
        tau = tau[0, :, 0, :]
        
        return self.compute_channel_metrics(a, tau)

    def compute_channel_metrics(
        self, a: np.ndarray, tau: np.ndarray
    ) -> Tuple[List[float], List[float], List[float]]:
        """Compute channel metrics from amplitude and delay data.

        Returns:
            Tuple containing delay spread list, mean delay list, and Rician K-factor list.
        """
        a_array = np.array(a)
        tau_array = np.array(tau)
        
        # Create mask for valid delay values (tau != -1)
        valid_paths_mask = tau_array != -1
        
        # Calculate power for all paths
        power = np.abs(a_array) ** 2
        
        # Mask out power values corresponding to invalid paths
        masked_power = power * valid_paths_mask
        
        # Calculate total power for valid paths only
        total_power = np.sum(masked_power, axis=1)
        
        # Create mask for channels with non-zero total power
        non_zero_mask = total_power > 0
        
        # Initialize output arrays
        mean_delay = np.zeros(len(a))
        delay_spread = np.zeros(len(a))
        
        # Compute mean delay only for valid paths
        mean_delay[non_zero_mask] = np.sum(
            tau_array[non_zero_mask] * masked_power[non_zero_mask], axis=1
        ) / total_power[non_zero_mask]
        
        # Compute delay spread
        tau_diff = tau_array - mean_delay[:, np.newaxis]
        
        # Mask tau_diff values corresponding to invalid paths
        masked_tau_diff = tau_diff * valid_paths_mask
        
        # Compute delay spread only for valid paths
        delay_spread[non_zero_mask] = np.sqrt(
            np.sum((masked_tau_diff[non_zero_mask]) ** 2 * masked_power[non_zero_mask], axis=1)
            / total_power[non_zero_mask]
        )

        # Compute Rician K-factor
        valid_mask = tau_array != -1  # Mask for valid paths
        tau_valid = np.where(valid_mask, tau_array, 1000)  # Replace -1 with 1000

        earliest_indices = np.argmin(tau_valid, axis=1)  # Indices of earliest paths

        P_LOS = power[np.arange(len(power)), earliest_indices]  # Power of earliest paths
        P_NLOS = total_power - P_LOS  # Power of other paths

        # Initialize K-factor array
        K = np.zeros(len(a))

        # Compute K-factor where total power is non-zero
        K[non_zero_mask] = np.divide(
            P_LOS[non_zero_mask],
            P_NLOS[non_zero_mask],
            out=np.full_like(P_LOS[non_zero_mask], np.nan),
            where=P_NLOS[non_zero_mask] != 0,
        )

        # Handle cases where total power is zero
        K[~non_zero_mask] = np.nan 

        return delay_spread.tolist(), mean_delay.tolist(), K.tolist()

    def reshape_results(
        self, delay_spreads: List[float], mean_excess_delays: List[float], Ks:List[Optional[float]], cell_centers_shape: Tuple[int, int], num_cells: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape results into final format."""
        cell_centers_shape_flat = (cell_centers_shape[0] * cell_centers_shape[1],)
        
        full_delay_spreads = np.full(cell_centers_shape_flat, np.nan)
        full_mean_excess_delays = np.full(cell_centers_shape_flat, np.nan)
        full_Ks = np.full(cell_centers_shape_flat, np.nan)
        
        full_delay_spreads[:len(delay_spreads)] = delay_spreads
        full_mean_excess_delays[:len(mean_excess_delays)] = mean_excess_delays
        full_Ks[:len(Ks)] = Ks
        
        return (
            full_delay_spreads.reshape(cell_centers_shape),
            full_mean_excess_delays.reshape(cell_centers_shape),
            full_Ks.reshape(cell_centers_shape)
        )

    def cleanup(self, local_vars: Dict[str, Any]) -> None:
        """Clean up resources and force garbage collection."""
        cleanup_vars = ['cm', 'tx', 'scene', 'full_delay_spreads',
                   'full_mean_excess_delays', 'delay_spreads',
                   'mean_excess_delays']
        for var in cleanup_vars:
            if var in local_vars:
                del local_vars[var]
        
        tf.keras.backend.clear_session()
        gc.collect()