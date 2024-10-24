from sionna.rt.radio_material import RadioMaterial
import numpy as np
import mitsuba as mi
import drjit as dr
import re
import random
import pandas as pd

def perturb_building_heights(scene, perturb_sigma):
    """
    Perturbs the height (z-coordinate) of the highest points of grouped objects in the scene by a random amount.
    Objects are grouped based on their names, ignoring the material part starting with "-itu".

    :param scene: A Scene object containing a list of 3D objects.
    """
    try:
        # Group objects by their base name (without material)
        building_groups = {}
        for key, obj in scene._scene_objects.items():
            name = obj._name
            base_name = re.split(r'-itu', name)[0]  # Split at '-itu' and take the first part
            if "terrain" not in base_name.lower() and "ground" not in base_name.lower():
                if base_name not in building_groups:
                    building_groups[base_name] = []
                building_groups[base_name].append(key)
        
        # Process each building group
        bldg_group_to_perturbation = {}
        for base_name, object_keys in building_groups.items():
            # Find the maximum z-coordinate across all objects in the group
            group_min_z = None
            for key in object_keys:
                obj = scene._scene_objects[key]
                mi_shape = obj._mi_shape
                params = mi.traverse(mi_shape)
                if 'vertex_positions' in params:
                    vertex_positions = params['vertex_positions']
                    vertex_positions = dr.unravel(mi.Point3f, vertex_positions)
                    obj_min_z = dr.min(vertex_positions.z)
                    if group_min_z is None:
                        group_min_z = obj_min_z
                    else:
                        group_min_z = dr.minimum(group_min_z, obj_min_z)

            # Generate a single random perturbation for the entire building group
            #perturbation = np.random.uniform(200, 300)
            perturbation = perturb_sigma * np.random.randn()
            # Apply the perturbation to all objects in the group
            for key in object_keys:
                obj = scene._scene_objects[key]
                mi_shape = obj._mi_shape
                params = mi.traverse(mi_shape)

                if 'vertex_positions' in params:
                    vertex_positions = params['vertex_positions']
                    vertex_positions = dr.unravel(mi.Point3f, vertex_positions)

                    # Create a mask for vertices at or very close to the maximum height
                    epsilon = 0.001
                    is_above_floor = (vertex_positions.z - group_min_z) > epsilon

                    # Apply perturbation only to the vertices at the maximum height
                    vertex_positions.z = dr.select(
                        is_above_floor,
                        dr.maximum(vertex_positions.z + perturbation, vertex_positions.z),
                        vertex_positions.z
                    )

                    if dr.sum(is_above_floor)>0:
                        bldg_group_to_perturbation[base_name] = perturbation
                    else:
                        bldg_group_to_perturbation[base_name] = 0

                    # Flatten vertex_positions back to original shape
                    vertex_positions = dr.ravel(vertex_positions)

                    # Update the vertex positions in the parameters
                    params['vertex_positions'] = vertex_positions
                    
                    # Update the shape parameters
                    mi_shape.parameters_changed()
                else:
                    print(f"Object '{obj._name}' does not have 'vertex_positions', skipping.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise  # Re-raise the exception for higher-level error handling
    
    return bldg_group_to_perturbation

def perturb_building_positions(scene, perturb_sigma):
    """
    Perturbs the x and y coordinates of grouped objects in the scene by a random amount.
    Objects are grouped based on their names, ignoring the material part starting with "_itu".

    :param scene: A Scene object containing a list of 3D objects.
    """
    bldg_group_to_perturbation = {}
    # Group objects by their base name (without material)
    building_groups = {}
    for key, obj in scene._scene_objects.items():
        name = obj._name
        base_name = re.split(r'-itu', name)[0]  # Split at '_itu' and take the first part
        if "terrain" not in base_name.lower() and "ground" not in base_name.lower():
            if base_name not in building_groups:
                building_groups[base_name] = []
            building_groups[base_name].append(key)
    
    # Process each building group
    for base_name, object_keys in building_groups.items():
        # Generate a single random perturbation for the entire building group
        perturbation_x = perturb_sigma * np.random.randn()
        perturbation_y = perturb_sigma * np.random.randn()
        bldg_group_to_perturbation[base_name] = [perturbation_x, perturbation_y]
        # Apply the perturbation to all objects in the group
        for key in object_keys:
            obj = scene._scene_objects[key]
            mi_shape = obj._mi_shape
            params = mi.traverse(mi_shape)

            if 'vertex_positions' in params:
                vertex_positions = params['vertex_positions']
                vertex_positions = dr.unravel(mi.Point3f, vertex_positions)

                # Apply perturbation to x and y coordinates
                vertex_positions.x += perturbation_x
                vertex_positions.y += perturbation_y

                # Flatten vertex_positions back to original shape
                vertex_positions = dr.ravel(vertex_positions)

                # Update the vertex positions in the parameters
                params['vertex_positions'] = vertex_positions
                
                # Update the shape parameters
                mi_shape.parameters_changed()
            else:
                print(f"Object '{obj._name}' does not have 'vertex_positions', skipping.")

    return bldg_group_to_perturbation

# Function to sample a material randomly considering their probabilities
def sample_material(probabilities=None):

    # List of materials and their corresponding probabilities
    materials = [
        "itu_concrete", "itu_brick", "itu_plasterboard", "itu_wood", 
        "itu_glass", "itu_ceiling_board", "itu_chipboard", "itu_plywood", 
        "itu_marble", "itu_metal", "itu_very_dry_ground", "itu_medium_dry_ground",
        "itu_wet_ground"
    ]

    if probabilities is None:
        probabilities = [
            0.25, 0.15, 0.1, 0.1, 
            0.05, 0.05, 0.03, 0.07, 
            0.05, 0.05, 0.02, 0.05,
            0.03
        ]

    return random.choices(materials, weights=probabilities, k=1)[0]

def perturb_material_properties(scene, material_types_known, verbose=False, **kwargs):
    """
    Perturb material properties for each object individually while maintaining original material types.
    
    Args:
        scene: Scene object containing materials and objects
        material_types_known (bool): Whether to perturb existing materials or sample new ones
        verbose (bool): Whether to print detailed information
        **kwargs: Additional arguments including rel_perm_sigma_ratio and cond_sigma_ratio
    
    Returns:
        dict: Dictionary mapping object names to their material perturbations
    """
    if verbose:
        for mat in list(scene.radio_materials.values()):
            if mat.is_used:
                print(f"Name: {mat.name}, used by {mat.use_counter} scene objects.")
    
    bldg_mat = {}
    
    if material_types_known:
        # Get perturbation ratios from kwargs or use defaults
        rel_perm_sigma_ratio = kwargs.get('rel_perm_sigma_ratio', 0.1)
        cond_sigma_ratio = kwargs.get('cond_sigma_ratio', 0.1)
        
        # Process each object individually
        for obj in scene.objects.values():
            original_material = obj.radio_material
            
            if not hasattr(original_material, 'conductivity'):
                # Skip objects without radio material properties
                continue
                
            # Calculate sigmas for this object's material
            cond_sigma = original_material.conductivity * cond_sigma_ratio
            rel_perm_sigma = original_material.relative_permittivity * rel_perm_sigma_ratio
            
            # Generate valid perturbations for this specific object
            cond_mod = -1.0
            while cond_mod < 0:
                cond_perturbation = cond_sigma * np.random.randn()
                cond_mod = original_material.conductivity + cond_perturbation
            
            rel_perm_mod = -1.0
            while rel_perm_mod <= 1.0:
                rel_perm_perturbation = rel_perm_sigma * np.random.randn()
                rel_perm_mod = original_material.relative_permittivity + rel_perm_perturbation
            
            # Create a unique modified material for this object    
            new_mat_name = f"{original_material.name}_mod_{obj.name.split('-itu')[0] if '-itu' in obj.name else obj.name}"
            new_mat = RadioMaterial(
                new_mat_name,
                relative_permittivity=rel_perm_mod,
                conductivity=cond_mod
            )
            scene.add(new_mat)
            
            # Store the perturbation information
            bldg_mat[obj.name] = {
                'material': original_material.name,
                'cond_perturbation': float(cond_perturbation),  # Convert to float if tensor
                'rel_perm_perturbation': float(rel_perm_perturbation)  # Convert to float if tensor
            }
            
            # Assign the new material to the object
            obj.radio_material = new_mat_name
            
            if verbose:
                print(f"\nObject: {obj.name}")
                print(f"Original material: {original_material.name}")
                print(f"New material: {new_mat_name}")
                print(f"Relative permittivity: {rel_perm_mod} (perturbation: {rel_perm_perturbation})")
                print(f"Conductivity: {cond_mod} (perturbation: {cond_perturbation})")
                
    elif material_types_known == 0:
        # Handle the case where material types are not known
        probabilities = kwargs.get('probabilities', None)
        
        for obj in scene.objects.values():
            obj.radio_material = sample_material(probabilities=probabilities)
            bldg_mat[obj.name] = {
                'material': obj.radio_material.name,
                'cond_perturbation': 0.0,
                'rel_perm_perturbation': 0.0
            }
    
    if verbose:
        print("\nFinal material usage summary:")
        for mat in list(scene.radio_materials.values()):
            if mat.is_used:
                print(f"Name: {mat.name}, used by {mat.use_counter} scene objects.")
    
    return bldg_mat

class PerturbationTracker:
    def __init__(self):
        """Initialize an empty DataFrame with the required structure."""
        self.df = pd.DataFrame(columns=[
            'scene_name',
            'perturbation_number',
            'tx_building',
            'tx_x',
            'tx_y',
            'tx_z',
            'building_name',
            'building_base_name',
            'material_name',
            'height_perturbation',
            'position_perturbation_x',
            'position_perturbation_y',
            'conductivity_perturbation',
            'relative_permittivity_perturbation'
        ])
        
    def _extract_base_name(self, building_name):
        """Extract base name from building name by removing -itu suffix."""
        return building_name.split('-itu')[0] if '-itu' in building_name else building_name
    
    def update(self, perturbation_number, tx_position, tx_building, 
              bldg_mat=None, bldg_group_to_height_pert=None, 
              bldg_group_to_pos_pert=None, scene_name="munich"):
        """
        Update the DataFrame with new perturbation data.
        
        Args:
            perturbation_number (int): Current perturbation iteration number
            tx_position (list/tuple): [x, y, z] coordinates of transmitter
            tx_building (str): Name of the building where TX is placed
            bldg_mat (dict): Dictionary containing material perturbations
            bldg_group_to_height_pert (dict): Dictionary containing height perturbations
            bldg_group_to_pos_pert (dict): Dictionary containing position perturbations
            scene_name (str): Name of the scene being simulated
        """
        # Create a set of all unique building base names
        building_bases = set()
        
        if bldg_mat:
            building_bases.update(self._extract_base_name(name) for name in bldg_mat.keys())
        if bldg_group_to_height_pert:
            building_bases.update(bldg_group_to_height_pert.keys())
        if bldg_group_to_pos_pert:
            building_bases.update(bldg_group_to_pos_pert.keys())
            
        # Create rows for each building
        new_rows = []
        for base_name in building_bases:
            row = {
                'scene_name': scene_name,
                'perturbation_number': perturbation_number,
                'tx_building': tx_building,
                'tx_x': tx_position[0],
                'tx_y': tx_position[1],
                'tx_z': tx_position[2],
                'building_base_name': base_name,
                'height_perturbation': None,
                'position_perturbation_x': None,
                'position_perturbation_y': None,
                'material_name': None,
                'conductivity_perturbation': None,
                'relative_permittivity_perturbation': None
            }
            
            # Add height perturbation if available
            if bldg_group_to_height_pert and base_name in bldg_group_to_height_pert:
                row['height_perturbation'] = bldg_group_to_height_pert[base_name]
                
            # Add position perturbation if available
            if bldg_group_to_pos_pert and base_name in bldg_group_to_pos_pert:
                row['position_perturbation_x'] = bldg_group_to_pos_pert[base_name][0]
                row['position_perturbation_y'] = bldg_group_to_pos_pert[base_name][1]
                
            # Add material perturbations if available
            if bldg_mat:
                # Find matching building names (there might be multiple materials per building)
                matching_bldgs = [name for name in bldg_mat.keys() 
                                if self._extract_base_name(name) == base_name]
                
                for bldg_name in matching_bldgs:
                    material_row = row.copy()
                    material_row['building_name'] = bldg_name
                    material_info = bldg_mat[bldg_name]
                    material_row['material_name'] = material_info['material']
                    material_row['conductivity_perturbation'] = material_info.get('cond_perturbation')
                    material_row['relative_permittivity_perturbation'] = material_info.get('rel_perm_perturbation')
                    new_rows.append(material_row)
                
            # If no material perturbations, add the basic row
            if not bldg_mat or not any(self._extract_base_name(name) == base_name for name in bldg_mat.keys()):
                new_rows.append(row)
        
        # Add new rows to DataFrame
        self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
    
    def get_dataframe(self):
        """Return the current state of the DataFrame."""
        return self.df
    
    def save_to_csv(self, filename):
        """Save the DataFrame to a CSV file."""
        self.df.to_csv(filename, index=False)
        
    def get_perturbation_summary(self, perturbation_number=None):
        """
        Get summary statistics for a specific perturbation or all perturbations.
        
        Args:
            perturbation_number (int, optional): Specific perturbation to summarize
        
        Returns:
            dict: Summary statistics
        """
        df = self.df if perturbation_number is None else \
             self.df[self.df['perturbation_number'] == perturbation_number]
        
        return {
            'total_buildings': df['building_base_name'].nunique(),
            'tx_location_summary': {
                'x_range': [df['tx_x'].min(), df['tx_x'].max()],
                'y_range': [df['tx_y'].min(), df['tx_y'].max()],
                'z_range': [df['tx_z'].min(), df['tx_z'].max()]
            },
            'avg_height_perturbation': df['height_perturbation'].mean(),
            'avg_position_perturbation': np.sqrt(
                df['position_perturbation_x'].pow(2) + 
                df['position_perturbation_y'].pow(2)
            ).mean(),
            'unique_materials': df['material_name'].nunique(),
            'avg_conductivity_perturbation': df['conductivity_perturbation'].mean(),
            'avg_permittivity_perturbation': df['relative_permittivity_perturbation'].mean()
        }
