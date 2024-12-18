import os
import json
import tempfile
import pandas as pd
import geopandas as gpd
import mercantile
import shapely
import trimesh
from tqdm import tqdm
from pyproj import CRS
from shapely import geometry

class BuildingFootprintProcessor:
    def __init__(self, min_lon, min_lat, max_lon, max_lat, data_dir):
        """
        Initialize the BuildingFootprintProcessor with AOI coordinates and data directory.
        """
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.data_dir = data_dir

        # Define AOI geometry
        self.aoi_geom = {
            "coordinates": [
                [
                    [self.min_lon, self.min_lat],
                    [self.min_lon, self.max_lat],
                    [self.max_lon, self.max_lat],
                    [self.max_lon, self.min_lat],
                    [self.min_lon, self.min_lat],
                ]
            ],
            "type": "Polygon",
        }
        self.aoi_shape = geometry.shape(self.aoi_geom)
        self.minx, self.miny, self.maxx, self.maxy = self.aoi_shape.bounds

        self.df_links = None  # DataFrame for dataset links
        self.quad_keys = None  # Set of quad keys covering the AOI
        self.combined_gdf = gpd.GeoDataFrame()
        self.scene_centroid = None

    def get_quad_keys(self, zoom=9):
        """
        Get the quad keys of tiles covering the AOI at a specified zoom level.
        """
        tiles = list(mercantile.tiles(self.minx, self.miny, self.maxx, self.maxy, zooms=zoom))
        self.quad_keys = {mercantile.quadkey(tile) for tile in tiles}
        print(f"The input area spans {len(self.quad_keys)} tiles: {self.quad_keys}")

    def load_dataset_links(self):
        """
        Load the dataset links from the CSV file.
        """
        self.df_links = pd.read_csv(
            "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv",
            dtype=str
        )

    def download_and_process_tiles(self):
        """
        Download GeoJSON files for each tile and merge them into a single GeoDataFrame.
        """
        idx = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_fns = []
            for quad_key in tqdm(self.quad_keys, desc="Processing QuadKeys"):
                rows = self.df_links[self.df_links["QuadKey"] == quad_key]
                if rows.shape[0] == 1:
                    url = rows.iloc[0]["Url"]
                    df_tile = pd.read_json(url, lines=True)
                    df_tile["geometry"] = df_tile["geometry"].apply(geometry.shape)
                    gdf_tile = gpd.GeoDataFrame(df_tile, crs=4326)
                    fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                    tmp_fns.append(fn)
                    if not os.path.exists(fn):
                        gdf_tile.to_file(fn, driver="GeoJSON")
                elif rows.shape[0] > 1: 
                    found_valid_data = False
                    for _, row in rows.iterrows():
                        url = row["Url"]
                        try:
                            df_tile = pd.read_json(url, lines=True)
                            df_tile["geometry"] = df_tile["geometry"].apply(geometry.shape)
                            gdf_tile = gpd.GeoDataFrame(df_tile, crs=4326)
                            # Apply the 'within' filter
                            gdf_tile = gdf_tile[gdf_tile.geometry.within(self.aoi_shape)]
                            if not gdf_tile.empty:
                                fn = os.path.join(tmpdir, f"{quad_key}.geojson")
                                tmp_fns.append(fn)
                                if not os.path.exists(fn):
                                    gdf_tile.to_file(fn, driver="GeoJSON")
                                found_valid_data = True
                                break  # Exit the loop over rows
                            else:
                                print(f"[INFO]: No geometries within AOI for URL {url} in quad_key {quad_key}. Trying next URL.")
                        except Exception as e:
                            print(f"[ERR]: Error processing URL {url} for quad_key {quad_key}: {e}")
                    if not found_valid_data:
                        print(f"[WARN]: No valid geometries found for quad_key {quad_key}.")
                else:
                    raise ValueError(f"[ERR]: QuadKey not found in dataset: {quad_key}")

            for fn in tmp_fns:
                gdf = gpd.read_file(fn)
                gdf = gdf[gdf.geometry.within(self.aoi_shape)]
                if gdf.empty:
                    print(f"No geometries within AOI in file {fn}.")
                    continue
                gdf['id'] = range(idx, idx + len(gdf))
                idx += len(gdf)
                self.combined_gdf = pd.concat([self.combined_gdf, gdf], ignore_index=True)

        self.combined_gdf = self.combined_gdf.to_crs('EPSG:4326')

    def create_ply_files(self):
        """
        Create PLY files from the building footprints.
        """
        df = self.combined_gdf.copy()
        if df.crs is None:
            df.crs = 'EPSG:4326'

        self.scene_centroid = df.geometry.unary_union.centroid
        lon_center = self.scene_centroid.x
        lat_center = self.scene_centroid.y

        aeqd_proj_string = (
            f"+proj=aeqd +lat_0={lat_center} +lon_0={lon_center} "
            "+x_0=0 +y_0=0 +units=m +ellps=WGS84 +no_defs"
        )
        aeqd_proj = CRS(aeqd_proj_string)
        df_projected = df.to_crs(aeqd_proj)

        meshes_dir = os.path.join(self.data_dir, "meshes")
        os.makedirs(meshes_dir, exist_ok=True)

        for idx, row in df_projected.iterrows():
            geom = row['geometry']
            if isinstance(geom, shapely.geometry.Polygon):
                polygons = [geom]
            elif isinstance(geom, shapely.geometry.MultiPolygon):
                polygons = list(geom)
            else:
                continue

            properties = row.get('properties', {})
            if isinstance(properties, str):
                properties = json.loads(properties)
            elif not isinstance(properties, dict):
                continue

            height = properties.get('height', 10)
            if height <= 0:
                print(f"[WARN]: Building at index {idx} has non-positive height ({height}). Skipping.")
                continue

            meshes = []
            for polygon in polygons:
                mesh = trimesh.creation.extrude_polygon(polygon, height)
                meshes.append(mesh)

            combined_mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            combined_mesh.visual.vertex_colors = [100, 100, 100, 255]
            ply_filename = f"{row['id']}.ply"
            combined_mesh.export(os.path.join(meshes_dir, ply_filename), file_type='ply')

    def process(self):
        """
        Run all processing steps.
        """
        self.get_quad_keys()
        self.load_dataset_links()
        self.download_and_process_tiles()
        self.create_ply_files()