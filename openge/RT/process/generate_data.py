import os
import numpy as np
import pandas as pd
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
from openge.RT.utils import lonlat_to_sionna_xy

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from sionna.rt.radio_material import RadioMaterial
import tensorflow as tf
import json
from datetime import datetime

class SimulationConfig:
    def __init__(self, config):
        self.config = config

    def save_config(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.config['save_dir'], f'config_{current_time}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as json_file:
            json.dump(self.config, json_file, indent=4)

class SimulationScene:
    def __init__(self, config):
        self.config = config
        self.scene = load_scene(os.path.join(self.config["geo_data_dir"], self.config["xml_filename"]))
        self.change_unk_materials()
        self.scene.frequency = self.config['freq']
        self.scene.synthetic_array = self.config['synthetic_array']
        self.configure_scene()

    def change_unk_materials(self, default_name = "itu_marble"):
        """
        Finds the highest z value at a given (x, y) pair in the scene.
        
        :param scene: A Scene object containing a list of 3D objects
        :param query_x: The x coordinate to query
        :param query_y: The y coordinate to query
        :return: The highest z value at the queried (x, y) position
        """
        invalid_rm_names = {}
        for key in self.scene._scene_objects.keys():
            name = self.scene._scene_objects[key]._name
            rm = self.scene._scene_objects[key]._radio_material.name
            if rm.startswith("itu") == 0:
                print(name)
                invalid_rm_names[key] = rm

        for key in invalid_rm_names.keys():
            mat = RadioMaterial(default_name)
            self.scene._scene_objects[key]._radio_material = mat

    def configure_scene(self):
        # Configure transmitter and receiver antenna arrays
        self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                          vertical_spacing=0.5, horizontal_spacing=0.5,
                                          pattern=self.config['tx_array_pattern'],
                                          polarization=self.config['tx_array_pol'])

        self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                          vertical_spacing=0.5, horizontal_spacing=0.5,
                                          pattern=self.config['rx_array_pattern'],
                                          polarization=self.config['rx_array_pol'])

        # Add a transmitter to the scene
        tx = Transmitter(name="tx", position=self.config['tx_loc'], orientation=[0, 0, 0])
        self.scene.add(tx)

        self.cm = self.scene.coverage_map(max_depth=self.config["max_depth"],
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=(1., 1.), # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(20e5)) # Reduce if your hardware does not have enough memory
    
    def find_highest_z_at_xy(self, query_x, query_y):
        """
        Finds the highest z value at a given (x, y) pair in the scene.
        
        :param scene: A Scene object containing a list of 3D objects
        :param query_x: The x coordinate to query
        :param query_y: The y coordinate to query
        :return: The highest z value at the queried (x, y) position
        """
        closest_z = None

        for key in self.scene._scene_objects.keys():
            name = self.scene._scene_objects[key]._name
            if "terrain" in name.lower():
                continue
            mi_shape = self.scene._scene_objects[key]._mi_shape

            face_indices3 = mi_shape.face_indices(dr.arange(mi.UInt32, mi_shape.face_count()))
            # Flatten. This is required for calling vertex_position
            # [n_prims*3]
            face_indices = dr.ravel(face_indices3)
            # Get vertices coordinates
            # [n_prims*3, 3]
            vertex_coords = mi_shape.vertex_position(face_indices)
            for vertex in np.array(vertex_coords):
                x, y, z = vertex
                distance = (x - query_x)**2 + (y - query_y)**2
                
                # Check if this vertex is closer than any previous vertex
                if closest_z is None:
                    closest_z = z
                if distance < 25:
                    closest_z = max(z, closest_z)
        return closest_z


class SimulationRunner:
    def __init__(self, config, verbose=False):
        self.config = config
        self.scene = SimulationScene(config)
        self.a = None
        self.tau = None
        self.rx_locs = []
        self.rx_vels = []
        self.verbose = verbose

    def run_simulation(self):
        num_runs = int(np.ceil(self.config['target_num_cirs'] / self.config['batch_size_cir']))
        for idx in range(num_runs):
            print(f"Progress: {idx+1}/{num_runs}", end="\r")
            self.simulate_batch()

        self.save_data(idx)
        return None
    
    def simulate_batch(self):
        # Sample random user positions
        ue_pos = self.scene.cm.sample_positions(
            batch_size=self.config['batch_size_cir'],
            min_gain_db=self.config['min_gain_db'],
            max_gain_db=self.config['max_gain_db'],
            min_dist=self.config['min_dist'],
            max_dist=self.config['max_dist'])

        for i in range(self.config['batch_size_cir']):
            rx = Receiver(name=f"rx-{i}", position=ue_pos[i])
            self.rx_locs.append(ue_pos[i])
            self.scene.scene.add(rx)

        # Simulate CIR for each receiver
        paths = self.scene.scene.compute_paths(
            max_depth=self.config['max_depth'],
            diffraction=True,
            edge_diffraction=self.config['edge_diffraction'],
            num_samples=self.config['num_samples'])

        # Transform paths into channel impulse responses
        paths.reverse_direction = False # downlink
        paths.normalize_delays = False
        rx_vel = [np.random.uniform(-40, 40), np.random.uniform(-40, 40), 0]
        paths.apply_doppler(sampling_frequency=self.config["subcarrier_spacing"],
                            num_time_steps=14,
                            tx_velocities=[0.,0.,0],
                            rx_velocities=rx_vel)
        
        if self.rx_vels is None:
            self.rx_vels = np.tile(rx_vel, (self.config["batch_size_cir"], 1))
        else:
            self.rx_vels.extend(np.tile(rx_vel, (self.config["batch_size_cir"], 1)))

        # We fix here the maximum number of paths to 100 which ensures
        # that we can simply concatenate different channel impulse reponses
        a_, tau_ = paths.cir(num_paths=100)

        del paths # Free memory

        if self.a is None:
            self.a = a_.numpy()
            self.tau = tau_.numpy()
        else:
            # Concatenate along the num_tx dimension
            self.a = np.concatenate([self.a, a_], axis=1)
            self.tau = np.concatenate([self.tau, tau_], axis=1)

        # Clear receivers from the scene after processing
        for i in range(self.config['batch_size_cir']):
            self.scene.scene.remove(f"rx-{i}")

        return None

    def map_to_grid(self, a, tau, delay_grid):
        # Initialize the output array with zeros
        a_mapped = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3], a.shape[4], self.config["delay_bins"], a.shape[6]), dtype=complex)
        
        # Loop over all batches and paths
        for batch in range(a.shape[1]):
            for path in range(tau.shape[3]):
                if tau[0, batch, 0, path] != -1:  # Valid path check
                    delay_value = tau[0, batch, 0, path]
                    # Find the closest index in the delay grid
                    grid_index = np.argmin(np.abs(delay_grid - delay_value))
                    # Assign or accumulate the complex amplitude to the nearest grid point
                    a_mapped[0, batch, 0, 0, 0, grid_index, :] += a[0, batch, 0, 0, 0, path, :]
                    
        return a_mapped

    def save_data(self, idx):
        p_link = np.sum(np.abs(self.a)**2, axis=(0,2,3,4,5,6))
        a_keep = self.a[0:1, p_link>0.,...]
        tau_keep = self.tau[0:1, p_link>0.,...]
        rx_locs_keep = np.array(self.rx_locs)[p_link>0, ...]
        rx_vels_keep = np.array(self.rx_vels)[p_link>0, ...]
        delay_grid = np.arange(self.config['delay_bins']) / 30.72e6
        a_mapped = self.map_to_grid(a_keep, tau_keep, delay_grid)

        a_tensor = tf.convert_to_tensor(a_mapped.transpose([1, 0, 2, 3, 4, 5, 6]), dtype=tf.float32)
        rx_locs_tensor = tf.convert_to_tensor(rx_locs_keep, dtype=tf.float32)
        rx_vels_tensor = tf.convert_to_tensor(rx_vels_keep, dtype=tf.float32)

        save_dir = self.config['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(os.path.join(save_dir, f"a_batch_{idx+1}.pt"))
        tf.io.write_file(os.path.join(save_dir, f"a_batch_{idx+1}.pt"), tf.io.serialize_tensor(a_tensor))
        tf.io.write_file(os.path.join(save_dir, f"rx_locs_batch_{idx+1}.pt"), tf.io.serialize_tensor(rx_locs_tensor))
        tf.io.write_file(os.path.join(save_dir, f"rx_vels_batch_{idx+1}.pt"), tf.io.serialize_tensor(rx_vels_tensor))

        print(f"[INFO]: Saved the channel, RX location, and RX velocity files successfully.")
        if self.verbose:
            plt.figure()
            plt.imshow(np.abs(a_mapped[0, 0, 0, 0, 0, :, :]).T, interpolation=None, cmap='hot', aspect='auto')
            plt.title(f"CIR Magnitude Spectrum for index {0}")
            plt.colorbar()
            plt.show()

        return None
