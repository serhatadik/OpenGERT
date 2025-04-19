from setuptools import setup, find_packages

setup(
    name='opengert',
    version='0.2.0',
    description='Open source geometry extraction and ray-tracing module.',
    author='Serhat Tadik',
    author_email='serhat.tadik@gatech.edu',
    packages=find_packages(),
    install_requires=[
        'bpy==4.2.0',
        'mitsuba==3.5.0',
        'pandas',
        'geopandas',
        'matplotlib',
        'seaborn',
        'sionna==0.19.0',
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
        'gdown',
        'ipykernel'
    ],
    extras_require={
        'gpu': [
            'nvidia-cudnn-cu12==8.9.2.26',
            'nvidia-cublas-cu12',
            'nvidia-cusolver-cu12',
            'nvidia-cufft-cu12',
            'nvidia-cusparse-cu12'
        ]
    },
    python_requires='~=3.11',
    scripts=[
        'scripts/call_ge.py',
        'scripts/call_rt.py',
        'scripts/run_montecarlo.py'
    ]
)
