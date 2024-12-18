from opengert.GE.ms.mesh_creation import TerrainMeshGenerator, BuildingFootprintProcessor
from opengert.GE.ms.to_xml import MeshesToSceneXML
from opengert.GE.osm import GeoExtentToSceneXML
import argparse
import os
from urllib.parse import urlparse

def main(source, blosm_path, mitsuba_blender_path, min_lon, min_lat, max_lon, max_lat, data_dir, export_filename, csv_source, csv_input):
    if source == 'osm':
        # Initialize GeoExtentToSceneXML
        geo = GeoExtentToSceneXML(
            data_dir=data_dir,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat
        )

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

    elif source == 'ms':
        # Initialize the TerrainMeshGenerator with CSV source information
        terrain_generator = TerrainMeshGenerator(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            data_dir=data_dir,
            csv_source=csv_source,
            csv_input=csv_input
        )

        # Run the TerrainMeshGenerator process
        terrain_generator.run()

        # Initialize the BuildingFootprintProcessor and run
        processor = BuildingFootprintProcessor(
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat,
            data_dir=data_dir
        )
        processor.process()

        # Initialize MeshesToSceneXML
        geo = MeshesToSceneXML(
            data_dir=data_dir,
            min_lon=min_lon,
            min_lat=min_lat,
            max_lon=max_lon,
            max_lat=max_lat
        )

        # Install and enable the mitsuba-blender addon
        geo.install_and_enable_addon(mitsuba_blender_path, module_name='mitsuba-blender')

        # Set up the Blender scene
        geo.setup_scene()

        geo.import_ply()

        geo.update_materials()

        # Export the scene to XML using mitsuba-blender
        geo.export_scene_to_xml(export_filename)

    else:
        raise ValueError("Invalid source specified. Please choose 'osm' or 'ms'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract geographical (terrain/building) data and export scene to XML.")

    # Add argument for source selection
    parser.add_argument("--source", type=str, choices=['osm', 'ms'], required=True, help="Data source to use: 'osm' or 'ms'.")

    # Add arguments for paths and parameters
    parser.add_argument("--blosm_path", type=str, default="./blosm_2.7.8.zip", help="Path to the blosm addon.")
    parser.add_argument("--mitsuba_blender_path", type=str, default="./mitsuba-blender.zip", help="Path to the mitsuba-blender addon.")

    # Coordinates for the region
    ## Georgia Tech
    parser.add_argument("--min_lon", type=float, default=-84.4072707707409, help="Minimum longitude.")
    parser.add_argument("--min_lat", type=float, default=33.77146527573862, help="Minimum latitude.")
    parser.add_argument("--max_lon", type=float, default=-84.38723383499998, help="Maximum longitude.")
    parser.add_argument("--max_lat", type=float, default=33.78140275118028, help="Maximum latitude.")
    
    ## Manhattan
    #parser.add_argument("--min_lon", type=float, default=-74.00131306747366, help="Minimum longitude.")
    #parser.add_argument("--min_lat", type=float, default=40.73777496711474, help="Minimum latitude.")
    #parser.add_argument("--max_lon", type=float, default=-73.97333548669778, help="Maximum longitude.")
    #parser.add_argument("--max_lat", type=float, default=40.76161926600843, help="Maximum latitude.")

    ## Seattle
    #parser.add_argument("--min_lon", type=float, default=-122.34459435993539, help="Minimum longitude.")
    #parser.add_argument("--min_lat", type=float, default=47.607976713824, help="Minimum latitude.")
    #parser.add_argument("--max_lon", type=float, default=-122.32861759376537, help="Maximum longitude.")
    #parser.add_argument("--max_lat", type=float, default=47.61861417931701, help="Maximum latitude.")

    # Data directory and export filename
    parser.add_argument("--data_dir", type=str, default="/home/gtpropagation/Documents/stadik/data/", help="Directory for data storage.")
    parser.add_argument("--export_filename", type=str, default="example_scene.xml", help="Filename for exporting the scene to XML.")

    # CSV source arguments
    parser.add_argument("--csv_source", type=str, choices=['url', 'local'], default='local',
                        help="Source of the CSV file: 'url' to download from a URL, 'local' to use a local file.")
    parser.add_argument("--url_csv_file", type=str, default="https://drive.google.com/uc?export=download&id=1tExtOHyIOnei0T4dO6wnaZXgydxYO79U",
                        help="URL or local file path to the DEM data CSV (required for 'ms' source).")

    args = parser.parse_args()

    # Validate arguments for 'ms' source
    if args.source == 'ms':
        if not args.url_csv_file:
            parser.error("--url_csv_file is required when source is 'ms'.")
        if args.csv_source == 'url':
            # Validate URL format
            parsed_url = urlparse(args.url_csv_file)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                parser.error("Invalid URL provided for --url_csv_file.")

    # Ensure the data directory exists
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"[INFO]: Created data directory at {args.data_dir}.")

    # Call the main function with parsed arguments
    main(
        source=args.source,
        blosm_path=args.blosm_path,
        mitsuba_blender_path=args.mitsuba_blender_path,
        min_lon=args.min_lon,
        min_lat=args.min_lat,
        max_lon=args.max_lon,
        max_lat=args.max_lat,
        data_dir=args.data_dir,
        export_filename=args.export_filename,
        csv_source=args.csv_source,
        csv_input=args.url_csv_file
    )
