from setuptools import setup, find_packages

setup(
    name='opengert',
    version='0.1.0',
    description='Open source geometry extraction and ray-tracing module.',
    author='Serhat Tadik',
    author_email='serhat.tadik@gatech.edu',
    packages=find_packages(),
    install_requires=[
        'bpy==3.6',
        'mitsuba==3.4.1',
        'pandas',
        'geopandas',
        'matplotlib',
        'sionna',
        'geopy',
        'mathutils',
        'mercantile',
        'shapely',
        'trimesh',
        'triangle',
        'tqdm',
        'pyproj',
        'rasterio',
        'plyfile',
        'requests',
    ],
    python_requires='~=3.10', 
    scripts=[
        'scripts/call_ge.py',
        'scripts/call_rt.py',
        'scripts/run_montecarlo.py'
    ]
)
