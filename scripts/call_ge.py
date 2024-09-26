from opengert.GE import GeoExtentToSceneXML
import argparse

def main(blosm_path, mitsuba_blender_path, min_lon, min_lat, max_lon, max_lat, data_dir, export_filename):
    geo = GeoExtentToSceneXML(data_dir=data_dir, min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)

    # Install and enable add-ons
    geo.install_and_enable_addon(blosm_path, module_name="blosm")
    geo.install_and_enable_addon(mitsuba_blender_path, module_name="mitsuba-blender")

    # Set up the Blender scene
    geo.setup_scene()

    # Import OSM data using blosm
    geo.import_blosm_data(data_type="terrain")
    geo.import_blosm_data(data_type="osm")

    geo.update_materials()

    # Export the scene to XML using mitsuba-blender
    geo.export_scene_to_xml(export_filename)

    geo.delete_remnants()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract geographical (terrain/building) data and export scene to XML.")

    # Add arguments for paths and parameters
    parser.add_argument("--blosm_path", type=str, default="./blosm_2.7.8.zip", help="Path to the blosm addon.")
    parser.add_argument("--mitsuba_blender_path", type=str, default="./mitsuba-blender.zip", help="Path to the mitsuba-blender addon.")

    # Coordinates for the region
    parser.add_argument("--min_lon", type=float, default=-84.40210587704061, help="Minimum longitude.")
    parser.add_argument("--min_lat", type=float, default=33.77401766838626, help="Minimum latitude.")
    parser.add_argument("--max_lon", type=float, default=-84.39545399832826, help="Maximum longitude.")
    parser.add_argument("--max_lat", type=float, default=33.78077739917423, help="Maximum latitude.")

    # Data directory and export filename
    parser.add_argument("--data_dir", type=str, default="/home/gtpropagation/Documents/stadik/data/", help="Directory for data storage.")
    parser.add_argument("--export_filename", type=str, default="example_scene.xml", help="Filename for exporting the scene to XML.")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.blosm_path,
        args.mitsuba_blender_path,
        args.min_lon,
        args.min_lat,
        args.max_lon,
        args.max_lat,
        args.data_dir,
        args.export_filename
    )
