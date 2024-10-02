import os
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from plyfile import PlyData, PlyElement
from pyproj import Proj, Transformer
import geopandas as gpd
from shapely.geometry import Point
import zipfile
import io
import os

class TerrainMeshGenerator:
    def __init__(self, min_lon, min_lat, max_lon, max_lat, data_dir,
                 csv_file='combined_dem_data_us.csv'):
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.tif_filepath = os.path.join(self.data_dir, "terrain.tif")
        self.ply_filepath = os.path.join(os.path.join(self.data_dir, "meshes"), "Terrain.ply")
        self.state_abbrev = None
        self.zone_number = None
        self.hemisphere = None
        self.easting_indices = None
        self.northing_indices = None
        self.final_url = None

    @staticmethod
    def download_and_extract_shapefile(url, extract_to='shapefile'):
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the download failed

        # Extract the zip file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extract_to)

        # Find the .shp file path within the extracted files
        for root, dirs, files in os.walk(extract_to):
            for file in files:
                if file.endswith('.shp'):
                    return os.path.join(root, file)
        
        raise FileNotFoundError("Shapefile (.shp) not found in the downloaded ZIP.")

    def get_state_abbreviation(self):
        # URL for the 2024 US states shapefile
        shapefile_url = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"
        
        # Download and extract the shapefile
        shapefile_path = self.download_and_extract_shapefile(shapefile_url)
    
        states = gpd.read_file(shapefile_path)
        center_lon = (self.min_lon + self.max_lon) / 2
        center_lat = (self.min_lat + self.max_lat) / 2
        point = Point(center_lon, center_lat)
        point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs='EPSG:4326')
        state_containing_point = gpd.sjoin(point_gdf, states, how='left', predicate='within')
        self.state_abbrev = state_containing_point.iloc[0]['STUSPS']
        print(f"[INFO]: State Abbreviation: {self.state_abbrev}")

    def get_utm_zone_and_hemisphere(self):
        ## Assumption: Min lon/lat is in the same zone/hemisphere as max lon/lat.
        if not -180.0 <= self.min_lon <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        if not -80.0 <= self.min_lat <= 84.0:
            raise ValueError("Latitude must be between -80 and 84 degrees.")
        self.zone_number = int((self.min_lon + 180) / 6) + 1
        self.hemisphere = 'N' if self.min_lat >= 0 else 'S'

    def get_easting_northing_indices(self):
        south = self.hemisphere == 'S'
        utm_proj = Proj(proj='utm', zone=self.zone_number, ellps='WGS84', south=south)
        easting_min, northing_min = utm_proj(self.min_lon, self.min_lat)
        easting_max, northing_max = utm_proj(self.max_lon, self.max_lat)
        self.easting_indices = np.arange(int(np.floor(easting_min / 10000)), int(np.floor(easting_max / 10000)) + 1)
        self.northing_indices = np.arange(int(np.ceil(northing_min / 10000)), int(np.ceil(northing_max / 10000)) + 1)
        print(f"[INFO]: Easting Indices: {self.easting_indices}")
        print(f"[INFO]: Northing Indices: {self.northing_indices}")

    def find_dem_url(self):
        df_url = pd.read_csv(self.csv_file, header=None).astype(str)
        tiff_column = df_url.apply(lambda col: col.str.contains('/TIFF/', na=False)).any()
        urls = df_url.loc[:, tiff_column]
        split_data = urls.iloc[:, 0].str.rsplit('/', n=1, expand=True)
        url_info = split_data.iloc[:, 1]
        substring = f'_x{self.easting_indices[0]}y{self.northing_indices[0]}_{self.state_abbrev}'
        filtered_series = url_info[url_info.str.contains(substring, case=True, na=False)]
        if not filtered_series.empty:
            self.final_url = urls.iloc[filtered_series.index[0], 0]
        else:
            raise ValueError(f"DEM URL not found for substring {substring}")

    def download_terrain_data(self):
        if os.path.exists(self.tif_filepath):
            print(f'[WARN]: File already exists at {self.tif_filepath}. Skipping download.')
            return  # Exit the function early if the file exists

        directory = os.path.dirname(self.tif_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        response = requests.get(self.final_url, stream=True)
        if response.status_code == 200:
            print(f'[INFO]: Downloading {self.final_url}...')
            with open(self.tif_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
        else:
            raise Exception(f'File not found at {self.final_url}')

    def generate_mesh(self):
        directory = os.path.dirname(self.ply_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with rasterio.open(self.tif_filepath) as dataset:
            dataset_crs = dataset.crs
            transformer = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
            min_x, min_y = transformer.transform(self.min_lon, self.min_lat)
            max_x, max_y = transformer.transform(self.max_lon, self.max_lat)
            row_min, col_min = dataset.index(min_x, min_y)
            row_max, col_max = dataset.index(max_x, max_y)
            row_min = max(0, min(row_min, dataset.height - 1))
            row_max = max(0, min(row_max, dataset.height - 1))
            col_min = max(0, min(col_min, dataset.width - 1))
            col_max = max(0, min(col_max, dataset.width - 1))
            row_start = min(row_min, row_max)
            row_stop = max(row_min, row_max) + 1
            col_start = min(col_min, col_max)
            col_stop = max(col_min, col_max) + 1
            window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            elevation = dataset.read(1, window=window)
            transform = dataset.window_transform(window)
            nodata = dataset.nodata
        height, width = elevation.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        x = transform.c + cols * transform.a + rows * transform.b
        y = transform.f + cols * transform.d + rows * transform.e
        center_row = height // 2
        center_col = width // 2
        x_center = transform.c + center_col * transform.a + center_row * transform.b
        y_center = transform.f + center_col * transform.d + center_row * transform.e
        x = x - x_center
        y = y - y_center
        x = x.flatten()
        y = y.flatten()
        z = elevation.flatten()
        mask = z != nodata
        x = x[mask]
        y = y[mask]
        z = z[mask]
        vertices = np.array([x, y, z]).T
        vertex_data = np.array(
            [tuple(v) for v in vertices],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        valid_mask = (elevation != nodata)
        valid_indices = np.full((height, width), -1)
        valid_indices[valid_mask] = np.arange(np.sum(valid_mask))
        if height > 1 and width > 1:
            ind_00 = valid_indices[:-1, :-1].flatten()
            ind_10 = valid_indices[1:, :-1].flatten()
            ind_01 = valid_indices[:-1, 1:].flatten()
            ind_11 = valid_indices[1:, 1:].flatten()
            mask_faces = (ind_00 != -1) & (ind_10 != -1) & (ind_01 != -1) & (ind_11 != -1)
            triangles1 = np.vstack((ind_00, ind_10, ind_01)).T[mask_faces]
            triangles2 = np.vstack((ind_10, ind_11, ind_01)).T[mask_faces]
            faces = np.vstack((triangles1, triangles2))
            faces_data = np.array(
                [(face,) for face in faces],
                dtype=[('vertex_indices', 'i4', (3,))]
            )
            vertex_element = PlyElement.describe(vertex_data, 'vertex')
            face_element = PlyElement.describe(faces_data, 'face')
            plydata = PlyData([vertex_element, face_element])
            plydata.write(self.ply_filepath)
            print(f"[INFO]: Conversion complete. PLY file saved as '{self.ply_filepath}'.")
        else:
            print("[ERR]: Not enough data points to create a mesh.")

    def run(self):
        self.get_state_abbreviation()
        self.get_utm_zone_and_hemisphere()
        self.get_easting_northing_indices()
        self.find_dem_url()
        self.download_terrain_data()
        self.generate_mesh()
