from setuptools import setup, find_packages

setup(
    name='opengert',
    version='0.1.0',
    description='Open source geometry extraction for ray-tracing module.',
    author='Serhat Tadik',
    author_email='serhat.tadik@gatech.edu',
    packages=find_packages(),
    install_requires=[
        'bpy==3.6',
        'mitsuba==3.4.1'
    ],
    python_requires='==3.10.11', 
    scripts=[
        'scripts/call_ge.py'
    ]
)
