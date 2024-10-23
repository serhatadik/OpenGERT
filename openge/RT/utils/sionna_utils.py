import geopy
import sionna.rt
from sionna.rt.radio_material import RadioMaterial
import numpy as np
import mitsuba as mi
import drjit as dr
import re
import matplotlib.pyplot as plt
import tensorflow as tf

def lonlat_to_sionna_xy(lat, lon, min_lon, min_lat, max_lon, max_lat):
    # Calculate the center of the scene
    center_scene_lat = (min_lat + max_lat) / 2.0
    center_scene_lon = (min_lon + max_lon) / 2.0

    # Define coordinates for latitude and longitude distance calculations
    coords_y1 = (lat, lon)
    coords_y2 = (center_scene_lat, lon)  # Point directly above or below on the same longitude

    coords_x1 = (lat, lon)
    coords_x2 = (lat, center_scene_lon)  # Point directly left or right on the same latitude

    # Calculate distances using geopy, but return signed distances
    distance_x = geopy.distance.distance(coords_x1, coords_x2).m
    distance_y = geopy.distance.distance(coords_y1, coords_y2).m

    # Adjust the sign of the distance based on the latitude and longitude difference
    if lat < center_scene_lat:
        distance_y = -distance_y  # Negative if latitude is south of the center
    if lon < center_scene_lon:
        distance_x = -distance_x  # Negative if longitude is west of the center

    return (distance_x, distance_y)

def change_unk_materials(scene, default_name = "itu_marble"):
    """
    Finds the highest z value at a given (x, y) pair in the scene.
    
    :param scene: A Scene object containing a list of 3D objects
    :param query_x: The x coordinate to query
    :param query_y: The y coordinate to query
    :return: The highest z value at the queried (x, y) position
    """
    invalid_rm_names = {}
    for key in scene._scene_objects.keys():
        name = scene._scene_objects[key]._name
        rm = scene._scene_objects[key]._radio_material.name
        if rm.startswith("itu") == 0:
            print(name)
            invalid_rm_names[key] = rm

    for key in invalid_rm_names.keys():
        mat = RadioMaterial(default_name)
        scene._scene_objects[key]._radio_material = mat

    return scene

def find_highest_z_at_xy(scene, query_x, query_y, include_ground=False):
    """
    Finds the highest z value and the corresponding object name at a given (x, y) pair in the scene.
    
    :param scene: A Scene object containing a list of 3D objects
    :param query_x: The x coordinate to query
    :param query_y: The y coordinate to query
    :param include_ground: Whether to include ground/terrain objects in the computation (default: False)
    :return: A tuple containing the highest z value and the name of the object it belongs to
    """
    closest_z = None
    object_name_with_highest_z = None

    for key in scene._scene_objects.keys():
        name = scene._scene_objects[key]._name
        if not include_ground:
            if "terrain" in name.lower() or "ground" in name.lower():
                continue

        mi_shape = scene._scene_objects[key]._mi_shape
        face_indices3 = mi_shape.face_indices(dr.arange(mi.UInt32, mi_shape.face_count()))
        # Flatten the indices for vertex extraction
        face_indices = dr.ravel(face_indices3)
        vertex_coords = mi_shape.vertex_position(face_indices)

        for vertex in np.array(vertex_coords):
            x, y, z = vertex
            distance = (x - query_x) ** 2 + (y - query_y) ** 2

            # Check if this vertex is closer than any previous vertex
            if closest_z is None or (distance < 25 and z > closest_z):
                closest_z = z
                object_name_with_highest_z = name

    return closest_z, object_name_with_highest_z

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

def perturb_material_properties(scene, rel_perm_sigma, cond_sigma, verbose=True):

    if verbose:
        for mat in list(scene.radio_materials.values()):
            if mat.is_used:
                print(f"Name: {mat.name}, used by {mat.use_counter} scene objects.")

    for mat in scene.radio_materials.values():
        if mat.is_used:        
            # Create new trainable material with some default values
            cond_mod = -1.0
            while cond_mod < 0:
                cond_mod = mat.conductivity + cond_sigma * np.random.randn()
            
            rel_perm_mod = -1.0
            while rel_perm_mod <= 1.0:
                rel_perm_mod = mat.relative_permittivity + rel_perm_sigma * np.random.randn()  

            new_mat = RadioMaterial(mat.name + "_mod",
                                    relative_permittivity = rel_perm_mod,
                                    conductivity = cond_mod )
            scene.add(new_mat)
            if verbose:
                print(f"New material name: {new_mat.name}")
                print(f"Relative permittivity: {rel_perm_mod}")
                print(f"Conductivity: {cond_mod}")

    # Assign trainable materials to the corresponding objects
    for obj in scene.objects.values():
        obj.radio_material = obj.radio_material.name + "_mod"

    if verbose:
        for mat in list(scene.radio_materials.values()):
            if mat.is_used:
                print(f"Name: {mat.name}, used by {mat.use_counter} scene objects.")

    return None