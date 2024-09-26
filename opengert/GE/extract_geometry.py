import bpy
import mitsuba
import shutil
import os

mitsuba.set_variant('scalar_rgb')

class GeoExtentToSceneXML():
    def __init__(self, data_dir, min_lon, min_lat, max_lon, max_lat):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.data_dir = data_dir

    def set_blosm_preferences(self):
        # Get addon preferences for blosm
        preferences = bpy.context.preferences
        addon_prefs = preferences.addons['blosm'].preferences

        # Set the directory for OSM data
        addon_prefs.dataDir = self.data_dir
        print(f"[INFO]: Set OSM data directory to: {self.data_dir}")

    def install_and_enable_addon(self, addon_path, module_name):
        # Install the addon from the provided path or zip file
        bpy.ops.preferences.addon_install(filepath=addon_path, overwrite=True)
        # Enable the installed addon using the correct module name
        bpy.ops.preferences.addon_enable(module=module_name)
        print(f"Installed and enabled {module_name}")
        if module_name=="blosm":
            self.set_blosm_preferences()

    @staticmethod
    def setup_scene():
        if "Cube" in bpy.data.objects or "Camera" in bpy.data.objects:
            # Set the 'Cube' object as the active object
            bpy.context.view_layer.objects.active = bpy.data.objects["Cube"]
            bpy.data.objects["Cube"].select_set(True)
            bpy.ops.object.delete()

        # Create a new world if it doesn't already exist
        if "World" not in bpy.data.worlds:
            bpy.ops.world.new()

        world = bpy.data.worlds['World']
        if world.node_tree:
            world.node_tree.nodes["Background"].inputs[0].default_value[2] = 0.97 # RGBA

        print("[INFO]: Scene setup completed!")

        return None

    @staticmethod
    def create_material(name, rgba):
        # Check if the material already exists
        mat = bpy.data.materials.get(name)
        if mat is None:
            # Create a new material
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True  # Enable nodes
            
            # Get the Principled BSDF node
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf is not None:
                bsdf.inputs['Base Color'].default_value = rgba  # Set RGBA color
        return mat

    def import_blosm_data(self, data_type):
        # Use blosm to import OpenStreetMap data within the given coordinates
        bpy.data.scenes["Scene"].blosm.maxLat = self.max_lat
        bpy.data.scenes["Scene"].blosm.maxLon = self.max_lon
        bpy.data.scenes["Scene"].blosm.minLat = self.min_lat
        bpy.data.scenes["Scene"].blosm.minLon = self.min_lon
        bpy.data.scenes["Scene"].blosm.dataType = data_type
        if data_type=="osm":
            blsm = bpy.data.scenes["Scene"].blosm
            blsm.water = False
            blsm.forests = False
            blsm.vegetation = False
            blsm.highways = False
            blsm.railways = False
            blsm.singleObject = False
        bpy.ops.blosm.import_data()

        return None
    
    def update_materials(
        self,
        metal_rgba=(0.2, 0.9, 0.8, 1.0), 
        marble_rgba=(1.0, 0.0, 0.3, 1.0), 
        terrain_rgba=(0.9, 0.9, 0.9, 1.0),
        default_rgba=(0.1, 0.1, 1.0, 1.0), 
        roof_material_name="itu_metal", 
        wall_material_name="itu_marble", 
        terrain_material_name="itu_concrete",
        default_material_name="itu_brick", 
    ):
        # Create materials with the specified RGBA values
        roof_material = self.create_material(roof_material_name, metal_rgba)
        wall_material = self.create_material(wall_material_name, marble_rgba)
        terrain_material = self.create_material(terrain_material_name, terrain_rgba)

        # Iterate over all objects in the scene
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':  # Check if the object is a mesh
                if obj.data.materials:
                    # Iterate over the object's materials and replace based on their names
                    for i, mat in enumerate(obj.data.materials):
                        if mat.name == "roof":
                            obj.data.materials[i] = roof_material
                        elif mat.name == "wall":
                            obj.data.materials[i] = wall_material
                        elif mat.name.lower() == "terrain":
                            obj.data.materials[i] == terrain_material
                else:
                    if default_material_name == "itu_metal":
                        default_material = roof_material
                    elif default_material_name == "itu_marble":
                        default_material = wall_material
                    elif default_material_name == "itu_concrete":
                        default_material = terrain_material
                    else:
                        default_material = self.create_material(default_material_name, default_rgba)
                    
                    # No materials, so assign the default material
                    obj.data.materials.append(default_material)

        print("[INFO]: Materials updated successfully!")
        return None

    
    def export_scene_to_xml(self, export_filename="example.xml"):
        # Assuming mitsuba-blender provides an export operator
        export_path = os.path.join(self.data_dir, export_filename)
        bpy.ops.export_scene.mitsuba(filepath=export_path, export_ids=True, axis_forward='Y', axis_up = 'Z', ignore_background=True)

        print(f"[INFO]: Scene exported to {export_path}")
        return None
    
    def delete_remnants(self):

        folder_paths = [os.path.join(self.data_dir, 'osm'), os.path.join(self.data_dir, 'terrain')] 
        for folder_path in folder_paths:
            try:
                shutil.rmtree(folder_path)
                print(f"[INFO]: Folder {folder_path} and its contents deleted successfully.")
            except OSError as e:
                print(f"Error: {e.strerror}")

        return None
