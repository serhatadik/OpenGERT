import geopy
from sionna.rt.radio_material import RadioMaterial
import numpy as np
import mitsuba as mi
import drjit as dr
import re
import random

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
    Finds the rooftop z coordinate of the building closest to a given (x, y) pair in the scene.
    
    :param scene: A Scene object containing a list of 3D objects
    :param query_x: The x coordinate to query
    :param query_y: The y coordinate to query
    :param include_ground: Whether to include ground/terrain objects in the computation (default: False)
    :return: A tuple containing the x, y of the query point, the highest z value at that point,
             and the name of the building it belongs to
    """
    closest_building_key = None
    minimal_distance = None

    # First pass: find the building whose vertex is closest to the query point
    for key in scene._scene_objects.keys():
        obj = scene._scene_objects[key]
        name = obj._name

        if not include_ground and ("terrain" in name.lower() or "ground" in name.lower()):
            continue

        mi_shape = obj._mi_shape
        face_indices = dr.ravel(mi_shape.face_indices(dr.arange(mi.UInt32, mi_shape.face_count())))
        vertex_coords = np.array(mi_shape.vertex_position(face_indices))

        # Find the minimal distance from any vertex in this object to the query point
        distances = np.sum((vertex_coords[:, :2] - np.array([query_x, query_y]))**2, axis=1)
        min_distance_in_obj = np.min(distances)

        if minimal_distance is None or min_distance_in_obj < minimal_distance:
            minimal_distance = min_distance_in_obj
            closest_building_key = key

    if closest_building_key is None:
        print("No building for the queried (x,y) coordinate was found. Exiting.")
        return query_x, query_y, None, None  # No building found

    # Second pass: within the closest building, find the highest z near the query point
    mi_shape = scene._scene_objects[closest_building_key]._mi_shape
    face_indices = dr.ravel(mi_shape.face_indices(dr.arange(mi.UInt32, mi_shape.face_count())))
    vertex_coords = np.array(mi_shape.vertex_position(face_indices))

    threshold_distance = 3.0  # Adjust this threshold based on scene scale
    distances = np.sum((vertex_coords[:, :2] - np.array([query_x, query_y]))**2, axis=1)
    close_vertices = vertex_coords[distances <= threshold_distance ** 2]

    if close_vertices.size > 0:
        highest_z = np.max(close_vertices[:, 2])
    else:
        # If no vertices are within the threshold, use the highest z in the building
        highest_z = np.max(vertex_coords[:, 2])

    object_name = scene._scene_objects[closest_building_key]._name
    return query_x, query_y, highest_z, object_name

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


def perturb_material_properties(scene, material_types_known, verbose=True, **kwargs):

    if verbose:
        for mat in list(scene.radio_materials.values()):
            if mat.is_used:
                print(f"Name: {mat.name}, used by {mat.use_counter} scene objects.")

    bldg_mat = {}
    if material_types_known:
        rel_perm_sigma_ratio, cond_sigma_ratio = 0.1, 0.1
        if 'rel_perm_sigma_ratio' in kwargs:
            rel_perm_sigma_ratio = kwargs['rel_perm_sigma_ratio']
        if 'cond_sigma_ratio' in kwargs:
            cond_sigma_ratio = kwargs['rel_perm_sigma_ratio']

        material_perturbation = {}

        for mat in scene.radio_materials.values():
            if mat.is_used:        
                # Create new trainable material with some default values
                cond_mod = -1.0
                cond_sigma = mat.conductivity * cond_sigma_ratio
                while cond_mod < 0:
                    cond_perturbation = cond_sigma * np.random.randn()
                    cond_mod = mat.conductivity + cond_perturbation

                rel_perm_mod = -1.0
                rel_perm_sigma = mat.relative_permittivity * rel_perm_sigma_ratio
                while rel_perm_mod <= 1.0:
                    rel_perm_perturbation = rel_perm_sigma * np.random.randn()
                    rel_perm_mod = mat.relative_permittivity + rel_perm_perturbation

                material_perturbation[mat.name] = {'material': mat.name, 'cond_perturbation': cond_perturbation.numpy(), 'rel_perm_perturbation': rel_perm_perturbation.numpy()}

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
            bldg_mat[obj.name] = material_perturbation[obj.radio_material.name]
            obj.radio_material = obj.radio_material.name + "_mod"
        if verbose:
            print(f"New material name: {new_mat.name}")
            print(f"Relative permittivity: {rel_perm_mod}")
            print(f"Conductivity: {cond_mod}")

    elif material_types_known==0:
        probabilities = None
        if 'probabilities' in kwargs:
            probabilities = kwargs['probabilities']
        for obj in scene.objects.values():
            obj.radio_material = sample_material(probabilities=probabilities)
            bldg_mat[obj.name] = {'material': obj.radio_material.name, 'cond_perturbation': 0.0, 'rel_perm_perturbation': 0.0}



    if verbose:
        for mat in list(scene.radio_materials.values()):
            if mat.is_used:
                print(f"Name: {mat.name}, used by {mat.use_counter} scene objects.")

    return bldg_mat