import multiprocessing
# Import TensorFlow and configure GPU settings
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Use all available GPUs
        tf.config.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using GPUs: {[gpu.name for gpu in logical_gpus]}")
        USE_GPU = True
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        USE_GPU = False
else:
    USE_GPU = False
    print("No GPU devices found. Running on CPU.")

import numpy as np
from opengert.RT.process import PerturbationSimulationManager

if __name__ == '__main__':
    from opengert.RT.utils import PerturbationTracker
    multiprocessing.set_start_method('spawn')  # Necessary for multiprocessing on some platforms

    # Initialize your tracker here
    tracker = PerturbationTracker()

    # Define a list of transmitter locations
    tx_locations = [
        [140, -160],
        [300, 300],
        [-500, 400],
    ]

    perturbation_config = {
        'use_gpu': USE_GPU,
        'analyze_chan_stats': True,
        'batch_size': 120 if USE_GPU else 20,
        'output_dir': "path_gain_results",
        'device': '/device:GPU:0' if USE_GPU else '/device:CPU:0',
        'num_perturbations': 4 if USE_GPU else 3,
        'tx_antenna_height': 6,
        'sim_material_perturbation': True,
        'sim_building_height_perturbation': True,
        'sim_building_position_perturbation': True,
        'verbose': False
    }

    # Initialize the PerturbationSimulationManager with the configuration and tracker
    simulation_manager = PerturbationSimulationManager(perturbation_config, tracker)

    # Process all TX locations
    for tx_xy in tx_locations:
        simulation_manager.process_tx_location(np.array(tx_xy))

    final_df = tracker.get_dataframe()
    print(final_df)
    tracker.save_to_csv('perturbation_results.csv')
    summary = tracker.get_perturbation_summary()
    print(summary)