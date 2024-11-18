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
    multiprocessing.set_start_method('spawn')

    # Initialize your tracker here
    #tracker = PerturbationTracker()

    # Define a list of transmitter locations
    tx_locations = np.random.uniform(-450, 450, (10, 2)).tolist()

    perturbation_config = {
        'scene_name': "munich",  # Enter the file name with xml extension here if scene is not readily available in Sionna RT.
        'use_gpu': USE_GPU,
        'analyze_chan_stats': True,
        'batch_size': 70 if USE_GPU else 20,
        'output_dir': "path_gain_results2",
        'device': '/device:GPU:0' if USE_GPU else '/device:CPU:0',
        'num_perturbations': 50 if USE_GPU else 30,
        'tx_antenna_height': 6,
        'sim_material_perturbation': False,
        'rel_perm_sigma_ratio': 0.1, # Standard deviation of Gaussian noise set to 10% of the value, creating relative noise scaling, only effective if the respective flag is set to True
        'cond_sigma_ratio': 0.1, # Standard deviation of Gaussian noise set to 10% of the value, creating relative noise scaling, only effective if the respective flag is set to True
        'sim_building_height_perturbation': True,
        'perturb_sigma_height': 1, # Only effective if the respective flag is set to True
        'sim_building_position_perturbation': False,
        'perturb_sigma_position': 0.4, # Only effective if the respective flag is set to True
        'verbose': False
    }

    # Initialize the PerturbationSimulationManager with the configuration and tracker
    simulation_manager = PerturbationSimulationManager(perturbation_config, tracker=None)

    # Process all TX locations
    for tx_xy in tx_locations:
        simulation_manager.process_tx_location(np.array(tx_xy))

    #final_df = tracker.get_dataframe()
    #print(final_df)
    #tracker.save_to_csv('/home/hice1/stadik3/OpenGERT/perturbation_results.csv')
    #summary = tracker.get_perturbation_summary()
    #print(summary)