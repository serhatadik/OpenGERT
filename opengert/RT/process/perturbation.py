import os
import sys
import gc
import re
import subprocess
import multiprocessing
from contextlib import nullcontext
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import tensorflow as tf

# Check if running in Google Colab
try:
    import google.colab
    colab_compat = True
except ImportError:
    colab_compat = False

# Import Mitsuba and configure variant based on CUDA availability
import mitsuba as mi

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
    PerturbationTracker
)


def process_perturbation_gpu(args) -> Dict[str, Any]:
    """Process a single perturbation on a specific GPU."""
    # Extract arguments
    (perturbation_index, gpu_id, tx_xy, tx_z, tx_bldg, basename,
     tx_dir, analyze_chan_stats, batch_size, sim_material_perturbation,
     sim_building_height_perturbation, sim_building_position_perturbation,
     verbose) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = '/device:GPU:0'

    if verbose:
        print(f"Processing perturbation {perturbation_index} on GPU {gpu_id}")

    # Create an instance of PerturbationProcessor
    processor = PerturbationProcessor(
        perturbation_index=perturbation_index,
        tx_xy=tx_xy,
        tx_z=tx_z,
        tx_bldg=tx_bldg,
        basename=basename,
        tx_dir=tx_dir,
        analyze_chan_stats=analyze_chan_stats,
        batch_size=batch_size,
        sim_material_perturbation=sim_material_perturbation,
        sim_building_height_perturbation=sim_building_height_perturbation,
        sim_building_position_perturbation=sim_building_position_perturbation,
        device=device,
        verbose=verbose
    )

    return processor.process_perturbation_core()


def process_perturbation_cpu(args) -> Dict[str, Any]:
    """Process a single perturbation for CPU parallel processing."""
    # Extract arguments
    (perturbation_index, tx_xy, tx_z, tx_bldg, basename,
     tx_dir, analyze_chan_stats, batch_size, sim_material_perturbation,
     sim_building_height_perturbation, sim_building_position_perturbation,
     verbose) = args

    if verbose:
        print(f"Starting perturbation {perturbation_index}")

    processor = PerturbationProcessor(
        perturbation_index=perturbation_index,
        tx_xy=tx_xy,
        tx_z=tx_z,
        tx_bldg=tx_bldg,
        basename=basename,
        tx_dir=tx_dir,
        analyze_chan_stats=analyze_chan_stats,
        batch_size=batch_size,
        sim_material_perturbation=sim_material_perturbation,
        sim_building_height_perturbation=sim_building_height_perturbation,
        sim_building_position_perturbation=sim_building_position_perturbation,
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
        tx_xy: np.ndarray,
        tx_z: float,
        tx_bldg: str,
        basename: str,
        tx_dir: str,
        analyze_chan_stats: bool,
        batch_size: int,
        sim_material_perturbation: bool,
        sim_building_height_perturbation: bool,
        sim_building_position_perturbation: bool,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        self.perturbation_index = perturbation_index
        self.tx_xy = tx_xy
        self.tx_z = tx_z
        self.tx_bldg = tx_bldg
        self.basename = basename
        self.tx_dir = tx_dir
        self.analyze_chan_stats = analyze_chan_stats
        self.batch_size = batch_size
        self.sim_material_perturbation = sim_material_perturbation
        self.sim_building_height_perturbation = sim_building_height_perturbation
        self.sim_building_position_perturbation = sim_building_position_perturbation
        self.device = device
        self.verbose = verbose

    def process_perturbation_core(self) -> Dict[str, Any]:
        """Core processing function shared between GPU and CPU implementations."""
        context = tf.device(self.device) if self.device else nullcontext()
        with context:
            # Load the scene
            scene = self.load_scene()
            # Apply perturbations
            pert_tx_xy, pert_tx_z, perturbation_data = self.apply_perturbations(scene)
            # Configure the scene
            self.configure_scene(scene, pert_tx_xy, pert_tx_z)
            # Process coverage maps and channel statistics
            result = self.process_coverage_and_stats(scene, pert_tx_xy, pert_tx_z)
            # Combine results
            result_dict = {
                'success': True,
                'perturbation_data': {
                    'perturbation_number': self.perturbation_index,
                    'tx_position': [pert_tx_xy[0], pert_tx_xy[1], pert_tx_z],
                    'tx_building': self.tx_bldg,
                    **perturbation_data
                },
                **result
            }
            # Clean up
            self.cleanup(locals())
            return result_dict

    def load_scene(self, scene_name=sionna.rt.scene.munich):
        """Load the simulation scene."""
        scene = load_scene(scene_name)
        return scene
    
    def apply_perturbations(self, scene) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Apply various perturbations to the scene."""

        bldg_mat = {}
        if self.sim_material_perturbation:
            bldg_mat = perturb_material_properties(scene, material_types_known=True, verbose=False)

        # Perturb building heights
        bldg_group_to_height_pert = {}
        pert_tx_z = float(self.tx_z)
        if self.sim_building_height_perturbation:
            perturb_sigma_height = 1
            bldg_group_to_height_pert = perturb_building_heights(scene, perturb_sigma=perturb_sigma_height)
            tx_z_perturb = bldg_group_to_height_pert[self.basename]
            pert_tx_z += tx_z_perturb

        # Perturb building positions
        bldg_group_to_pos_pert = {}
        pert_tx_xy = np.array(self.tx_xy, dtype=np.float64)
        if self.sim_building_position_perturbation:
            perturb_sigma_position = 0.4
            bldg_group_to_pos_pert = perturb_building_positions(scene, perturb_sigma=perturb_sigma_position)
            tx_xy_perturb = bldg_group_to_pos_pert[self.basename]
            pert_tx_xy += np.array(tx_xy_perturb)

        perturbation_data = {
            'bldg_mat': bldg_mat if self.sim_material_perturbation else None,
            'bldg_group_to_height_pert': bldg_group_to_height_pert if self.sim_building_height_perturbation else None,
            'bldg_group_to_pos_pert': bldg_group_to_pos_pert if self.sim_building_position_perturbation else None
        }

        return pert_tx_xy, pert_tx_z, perturbation_data

    def configure_scene(self, scene, pert_tx_xy: np.ndarray, pert_tx_z: float) -> None:
        """Configure antenna arrays and transmitter in the scene."""
        scene.tx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.7,
            horizontal_spacing=0.5,
            pattern="tr38901",
            polarization="V"
        )
        
        scene.rx_array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="dipole",
            polarization="V"
        )
        
        tx = Transmitter(
            name="tx",
            position=[pert_tx_xy[0], pert_tx_xy[1], pert_tx_z],
            orientation=[0, 0, 0]
        )
        scene.add(tx)

    def process_coverage_and_stats(
        self, scene, pert_tx_xy: np.ndarray, pert_tx_z: float
    ) -> Dict[str, Any]:
        """Process coverage maps and channel statistics."""
        context = tf.device(self.device) if self.device else nullcontext()
        
        with context:
            # Initial coverage map
            cm = scene.coverage_map(
                max_depth=7,
                los=True,
                reflection=True,
                diffraction=True,
                ris=False,
                cm_cell_size=(5, 5),
                num_samples=int(6e5),
                num_runs=1
            )
            
            cm_center = cm.center
            path_gain = cm.path_gain.numpy()[0, :, :]
            
            # Save path gain
            tx_z_formatted = float(f"{pert_tx_z:.3f}")
            np_filename = f"tx_{pert_tx_xy[0]}_{pert_tx_xy[1]}_z_{tx_z_formatted}_perturb_{self.perturbation_index}_path_gain.npy"
            np_filepath = os.path.join(self.tx_dir, np_filename)
            np.save(np_filepath, path_gain)
            
            channel_stats = None
            cm2_center = None
            cm2_cell_size = None
            
            if self.analyze_chan_stats:
                channel_stats, cm2_center, cm2_cell_size = self.process_channel_statistics(scene)
        
        return {
            'path_gain': path_gain,
            'grid_origin': {
                'path_gain': cm_center,
                'channel_stats': cm2_center
            },
            'cm_cell_size': {
                'path_gain': (5, 5),
                'channel_stats': cm2_cell_size
            },
            'channel_stats': channel_stats
        }

    def process_channel_statistics(
        self, scene
    ) -> Tuple[Dict[str, np.ndarray], Any, Tuple[int, int]]:
        """Process detailed channel statistics."""
        cm2_cell_size = (15, 15)
        context = tf.device(self.device) if self.device else nullcontext()
        
        with context:
            cm2 = scene.coverage_map(
                max_depth=7,
                los=True,
                reflection=True,
                diffraction=True,
                ris=False,
                cm_cell_size=cm2_cell_size,
                num_samples=int(6e5),
                num_runs=1
            )
            
        cell_centers = cm2.cell_centers.numpy()
        cell_centers_flat = cell_centers.reshape(-1, 3)
        cell_centers_shape = cell_centers.shape[:2]
        
        delay_spreads, mean_excess_delays = self.process_batches(
            scene, self.batch_size, cell_centers_flat, cell_centers_shape, cm2.orientation
        )
        
        return (
            {
                'delay_spreads': delay_spreads,
                'mean_excess_delays': mean_excess_delays
            },
            cm2.center,
            cm2_cell_size
        )

    def process_batches(
        self, scene, batch_size, cell_centers_flat: np.ndarray, cell_centers_shape: Tuple[int, int], orientation: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process batches of receivers for channel statistics."""
        delay_spreads = []
        mean_excess_delays = []
        #num_cells = len(cell_centers_flat)
        num_cells = 320
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
            delay_spread, mean_delay = self.process_single_batch(scene)
            
            # Append results
            delay_spreads.extend(delay_spread)
            mean_excess_delays.extend(mean_delay)
            
            # Cleanup
            for idx in range(len(batch_centers)):
                scene.remove(f"rx_{idx}")
            gc.collect()
        
        # Reshape results
        return self.reshape_results(
            delay_spreads,
            mean_excess_delays,
            cell_centers_shape,
            num_cells
        )

    def process_single_batch(self, scene) -> Tuple[List[float], List[float]]:
        """Process a single batch of receivers."""
        paths = scene.compute_paths(
            max_depth=7,
            num_samples=int(6e5),
            los=True,
            reflection=True,
            diffraction=True,
            scattering=False,
            ris=False,
            edge_diffraction=True
        )
        paths.normalize_delays = False
        a, tau = paths.cir(num_paths=50)
        del paths
        
        a = a[0, :, 0, 0, 0, :, 0]
        tau = tau[0, :, 0, :]
        
        return self.compute_channel_metrics(a, tau)

    def compute_channel_metrics(
        self, a: np.ndarray, tau: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Compute channel metrics from amplitude and delay data."""
        a_array = np.array(a)
        tau_array = np.array(tau)
        
        power = np.abs(a_array)**2
        total_power = np.sum(power, axis=1)
        
        non_zero_mask = total_power > 0
        mean_delay = np.zeros(len(a))
        delay_spread = np.zeros(len(a))
        
        mean_delay[non_zero_mask] = np.sum(
            tau_array[non_zero_mask] * power[non_zero_mask], axis=1
        ) / total_power[non_zero_mask]
        
        tau_diff = tau_array - mean_delay[:, np.newaxis]
        delay_spread[non_zero_mask] = np.sqrt(
            np.sum((tau_diff[non_zero_mask])**2 * power[non_zero_mask], axis=1)
            / total_power[non_zero_mask]
        )
        
        return delay_spread.tolist(), mean_delay.tolist()

    def reshape_results(
        self, delay_spreads: List[float], mean_excess_delays: List[float], cell_centers_shape: Tuple[int, int], num_cells: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reshape results into final format."""
        cell_centers_shape_flat = (cell_centers_shape[0] * cell_centers_shape[1],)
        
        full_delay_spreads = np.full(cell_centers_shape_flat, np.nan)
        full_mean_excess_delays = np.full(cell_centers_shape_flat, np.nan)
        
        full_delay_spreads[:num_cells] = delay_spreads
        full_mean_excess_delays[:num_cells] = mean_excess_delays
        
        return (
            full_delay_spreads.reshape(cell_centers_shape),
            full_mean_excess_delays.reshape(cell_centers_shape)
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


class PerturbationSimulationManager:
    def __init__(self, perturbation_config, tracker):
        self.perturbation_config = perturbation_config
        self.tracker = tracker

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

        analyze_chan_stats = perturbation_config["analyze_chan_stats"]
        batch_size = perturbation_config["batch_size"]
        output_dir = perturbation_config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        device = perturbation_config["device"]
        tx_antenna_height = perturbation_config["tx_antenna_height"]
        num_perturbations = perturbation_config["num_perturbations"]
        sim_material_perturbation = perturbation_config["sim_material_perturbation"]
        sim_building_height_perturbation = perturbation_config["sim_building_height_perturbation"]
        sim_building_position_perturbation = perturbation_config["sim_building_position_perturbation"]
        verbose = perturbation_config["verbose"]
        

        # Get initial TX position
        scene = load_scene(sionna.rt.scene.munich) 
        tx_x, tx_y, tx_z, tx_bldg = find_highest_z_at_xy(scene, tx_xy[0], tx_xy[1], include_ground=False)
        basename = re.split(r'-itu', tx_bldg)[0]
        tx_xy = [tx_x, tx_y]
        tx_z += tx_antenna_height
        print(f"tx_xy: {tx_xy}")
        print(f"tx_z: {tx_z}")
        del scene
        tx_dir = os.path.join(output_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}")
        os.makedirs(tx_dir, exist_ok=True)

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
                    tx_xy,
                    tx_z,
                    tx_bldg,
                    basename,
                    tx_dir,
                    analyze_chan_stats,
                    batch_size,
                    sim_material_perturbation,
                    sim_building_height_perturbation,
                    sim_building_position_perturbation,
                    verbose
                )
                for pert_idx in range(num_perturbations)
            ]
            process_func = process_perturbation_gpu
        else:
            num_workers = 4 # Use all available CPU cores
            print(f"Number of CPU cores used: {num_workers}")
            tasks = [
                (
                    pert_idx,
                    tx_xy,
                    tx_z,
                    tx_bldg,
                    basename,
                    tx_dir,
                    analyze_chan_stats,
                    batch_size,
                    sim_material_perturbation,
                    sim_building_height_perturbation,
                    sim_building_position_perturbation,
                    verbose
                )
                for pert_idx in range(num_perturbations)
            ]
            process_func = process_perturbation_cpu

        # Run perturbations
        results, path_gain_list, med_list, ds_list, grid_origin, cm_cell_size, grid_origin2, cm_cell_size2 = \
            self.run_perturbations(num_workers, tasks, process_func, tx_xy)

        if not path_gain_list:
            print("No successful perturbations. Exiting.")
            return
        else:
            # Process path_gain_list
            self.process_path_gain_results(path_gain_list, grid_origin, cm_cell_size, tx_xy, tx_dir)

        if med_list and ds_list:
            self.process_channel_stats_results(med_list, 'med', grid_origin2, cm_cell_size2, tx_xy, tx_dir)
            self.process_channel_stats_results(ds_list, 'ds', grid_origin2, cm_cell_size2, tx_xy, tx_dir)

    def run_perturbations(self, num_workers, tasks, process_func, tx_xy):
        """Run perturbations using multiprocessing."""
        results = []
        path_gain_list = []
        med_list = []
        ds_list = []
        grid_origin = None
        cm_cell_size = None
        grid_origin2 = None
        cm_cell_size2 = None

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_func, task_args) for task_args in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing TX at {tx_xy}'):
                result = future.result()
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
                    self.tracker.update(**result['perturbation_data'])
                    if self.perturbation_config["verbose"]:
                        print(f"Completed perturbation {perturbation_number}")
                else:
                    print(f"Perturbation {result['perturbation_data']['perturbation_number']} failed.")

        # Sort results by perturbation index to maintain order
        results.sort(key=lambda x: x['perturbation_data']['perturbation_number'])

        return results, path_gain_list, med_list, ds_list, grid_origin, cm_cell_size, grid_origin2, cm_cell_size2

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
        
        # Replace broken links with nan for statistics calculation
        path_gain_masked = stacked_path_gain.copy()
        path_gain_masked[broken_links_mask] = np.nan
        
        # Convert to dB, ignoring nan values
        path_gain_array_db = 10 * np.log10(path_gain_masked)
        
        # Calculate statistics (nan values will be automatically excluded)
        path_gain_mean_db = np.nanmean(path_gain_array_db, axis=0)
        path_gain_std_db = np.nanstd(path_gain_array_db, axis=0)
        
        # Save results including broken links frequency
        mean_filename = os.path.join(tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_mean.npy")
        std_filename = os.path.join(tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_std.npy")
        freq_filename = os.path.join(tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_broken_links_freq.npy")
        np.save(mean_filename, path_gain_mean_db)
        np.save(std_filename, path_gain_std_db)
        np.save(freq_filename, broken_links_frequency)
        
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
            path_gain_mean_db,
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label='Mean Path Gain (dB)')
        plt.title(f"Mean Path Gain for TX at {tx_xy}\n(Excluding Broken Links)")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        mean_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_mean.png")
        plt.savefig(mean_fig_filename)
        plt.close()
        
        # Plot standard deviation of path gain
        plt.figure(figsize=(8, 6))
        plt.imshow(
            path_gain_std_db,
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label='Std Dev of dB Path Gain')
        plt.title(f"Std Dev of Path Gain for TX at {tx_xy}\n(Excluding Broken Links)")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        std_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_path_gain_std.png")
        plt.savefig(std_fig_filename)
        plt.close()
        
        # Plot broken links frequency
        plt.figure(figsize=(8, 6))
        plt.imshow(
            broken_links_frequency,
            extent=extent,
            origin='lower',
            cmap='YlOrRd',  # Different colormap to distinguish from other plots
            aspect='auto')
        plt.colorbar(label='Number of Broken Links')
        plt.title(f"Broken Links Frequency for TX at {tx_xy}")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        freq_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_broken_links_freq.png")
        plt.savefig(freq_fig_filename)
        plt.close()
        
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
        stat_mean = np.mean(stat_stck, axis=0)
        stat_std = np.std(stat_stck, axis=0)

        # Save results
        mean_filename = os.path.join(tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_{stat_name}_mean.npy")
        std_filename = os.path.join(tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_{stat_name}_std.npy")
        np.save(mean_filename, stat_mean)
        np.save(std_filename, stat_std)
        
        # Get coverage map extent for plotting
        extent = (
            grid_origin[0] - cm_cell_size[0] * stat_mean.shape[1] / 2,
            grid_origin[0] + cm_cell_size[0] * stat_mean.shape[1] / 2,
            grid_origin[1] - cm_cell_size[1] * stat_mean.shape[0] / 2,
            grid_origin[1] + cm_cell_size[1] * stat_mean.shape[0] / 2
        )
        
        # Plot mean
        plt.figure(figsize=(8, 6))
        plt.imshow(
            stat_mean,
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label=f'Mean {stat_name.upper()}')
        plt.title(f"Mean {stat_name.upper()} for TX at {tx_xy}")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        mean_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_{stat_name}_mean.png")
        plt.savefig(mean_fig_filename)
        plt.close()
        
        # Plot standard deviation
        plt.figure(figsize=(8, 6))
        plt.imshow(
            stat_std,
            extent=extent,
            origin='lower',
            cmap='hot',
            aspect='auto')
        plt.colorbar(label=f'Std Dev of {stat_name.upper()}')
        plt.title(f"Std Dev of {stat_name.upper()} for TX at {tx_xy}")
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        std_fig_filename = os.path.join(
            tx_dir, f"tx_{tx_xy[0]}_{tx_xy[1]}_{stat_name}_std.png")
        plt.savefig(std_fig_filename)
        plt.close()
        
        # Clean up
        del stat_stck
        del stat_mean
        del stat_std
        del stat_list
        gc.collect()
