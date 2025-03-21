# OpenGERT: Open Source Automated Geometry Extraction with Geometric and Electromagnetic Sensitivity Analyses for Ray-Tracing Propagation Models

| ![Seattle](figures/seattle3.png) | ![Georgia Tech - Atlanta](figures/georgia_tech.png) | ![Manhattan](figures/manhattan.png) |
| :---: | :---: | :---: |
| **Seattle** | **Georgia Tech - Atlanta** | **Manhattan** |

*Seattle, Georgia Tech, and Manhattan represented in Sionna RT through automatic geometry extraction pipeline.*

## How to Begin
### **Function 1: Automated Geometry Extraction**

If you need to perform automated geometry extraction, follow these steps:

1. **Install Blender Software (Version 3.6)**
   
   - **Download Blender 3.6:**
     - Visit the [official Blender download page](https://www.blender.org/download/releases/3-6/) to download Blender version 3.6.
     - Choose the appropriate installer for your operating system and follow the installation instructions.

2. **Download and Install Required Blender Add-ons**

   - **Blosm 2.7.8**

   - **Mitsuba-Blender Add-on (Version 0.3.0 or Nightly Release):**
     - **Download Link:** [Mitsuba-Blender Releases](https://github.com/mitsuba-renderer/mitsuba-blender/releases)
     - **Important:** *Do **not** use Mitsuba-Blender v0.4.0, as it is **not** compatible and will not work.*


3. **Clone the OpenGERT Repository**

   ```bash
   git clone https://github.com/serhatadik/OpenGERT.git
   ```

4. **Set Up a Virtual Environment and Install OpenGERT**

   - **Create a Virtual Environment:**
     ```bash
     python3 -m venv venv
     ```
     *This command creates a virtual environment named `venv`.*

   - **Activate the Virtual Environment:**

     - **On macOS/Linux:**
       ```bash
       source venv/bin/activate
       ```
     - **On Windows:**
       ```bash
       venv\Scripts\activate
       ```
   - **Install the OpenGERT Package:**
     ```bash
     cd OpenGERT
     pip install -e .
     ```
     *Installs OpenGERT in editable mode.*

5. **Verify the Installation**

   - Ensure that the OpenGERT package is installed correctly by running:
     ```bash
     pip list
     ```
     *You should see `OpenGERT` listed among the installed packages.*

6. **Run the Geometry Extraction Script**
   
   Once you have installed Blender and the required add-ons, you can extract geometry by `cd`ing into OpenGERT directory and running the `scripts/call_ge.py` script. This script requires command-line arguments to specify:
   
   - The source of geometry extraction: **`osm`** or **`ms`**  
   - The minimum and maximum longitude: **`min_lon`** and **`max_lon`**  
   - The minimum and maximum latitude: **`min_lat`** and **`max_lat`**
   - The blosm and mitsuba_blender paths: **`blosm_path`** and **`mitsuba_blender_path`**
   - The data directory in which the scene files will be stored: **`data_dir`**
   - The xml filename that will represent the scene: **`export_filename`**

#### **Usage Example**
   
   ```bash
   python scripts/call_ge.py --source osm \
                             --min_lon -84.4072707707409 --max_lon -84.38723383499998 \
                             --min_lat 33.77146527573862  --max_lat 33.78140275118028 \
                             --blosm_path C:\Users\serha\blosm_2.7.8.zip \
                             --mitsuba_blender_path C:\Users\serha\mitsuba-blender.zip \
                             --data_dir C:\Users\serha\OpenGERT\data\gatech \
                             --export_filename "gt.xml"
   ```
   
   The script will automatically extract the relevant geometry from the chosen source and prepare it for further processing.
### **Function 2: Sensitivity Analysis Using Pre-Extracted Scenes**

If you already have a `.xml` scene file and meshes ready (or prefer to use the pre-existing Munich or Etoile scenes in Sionna RT without performing geometry extraction), follow these steps:

1. **Clone the OpenGERT Repository**

   ```bash
   git clone https://github.com/serhatadik/OpenGERT.git
   ```

2. **Set Up a Virtual Environment and Install OpenGERT**

   - **Create a Virtual Environment:**
     ```bash
     python3 -m venv venv
     ```
     *This command creates a virtual environment named `venv`.*

   - **Activate the Virtual Environment:**

     - **On macOS/Linux:**
       ```bash
       source venv/bin/activate
       ```
     - **On Windows:**
       ```bash
       venv\Scripts\activate
       ```
   - **Install the OpenGERT Package:**
     ```bash
     cd OpenGERT
     pip install -e .
     ```
     *Installs OpenGERT in editable mode.*

3. **Verify the Installation**

   - Ensure that the OpenGERT package is installed correctly by running:
     ```bash
     pip list
     ```
     *You should see `OpenGERT` listed among the installed packages.*

4. **Configure and Run the Perturbation Analysis**

   To conduct a perturbation analysis, open the file `scripts/run_montecarlo.py` and modify the `perturbation_config` dictionary according to your needs. Below is an example configuration with some default values:

   ```python
   perturbation_config = {
       'scene_name': "munich",  # Enter the file name with .xml extension if not using a default Sionna RT scene.
       'use_gpu': USE_GPU,
       'analyze_chan_stats': True,
       'batch_size': 70 if USE_GPU else 20,
       'output_dir': "results_munich_height_pert",
       'device': '/device:GPU:0' if USE_GPU else '/device:CPU:0',
       'num_perturbations': 50 if USE_GPU else 30,
       'tx_antenna_height': 6,
       'sim_material_perturbation': False,
       'rel_perm_sigma_ratio': 0.1,  # Effective only if 'sim_material_perturbation' = True
       'cond_sigma_ratio': 0.1,      # Effective only if 'sim_material_perturbation' = True
       'sim_building_height_perturbation': True,
       'perturb_sigma_height': 1,    # Effective only if 'sim_building_height_perturbation' = True
       'sim_building_position_perturbation': False,
       'perturb_sigma_position': 0.4, # Effective only if 'sim_building_position_perturbation' = True
       'verbose': False
   }
   ```

   After adjusting any parameters (e.g., scene name, number of perturbations, whether to use height or position perturbations, etc.), run:

   ```bash
   python scripts/run_montecarlo.py
   ```

   This script will apply the specified perturbations to your scene and produce results in the directory you specified (e.g., `results_munich_height_pert`).

### **Additional Tips**

- **Troubleshooting:**
  - If you encounter issues during installation or setup, refer to the [OpenGERT Issues Page](https://github.com/serhatadik/OpenGERT/issues) to seek solutions or report bugs.


## Geometry Extraction Workflow

![Workflow Diagram](figures/ge_pipeline.png)

## Sensitivity Analysis

Below are the detailed results of our sensitivity analysis conducted on the Etoile scene with height perturbations:

| ![Path Gain Std](figures/tx_60.6_149.6_path_gain_std.png) | ![Link Outage Freq](figures/tx_60.6_149.6_broken_links_freq.png) |
| :---: | :---: |
| **Path Gain Standard Deviation, Height Perturbation, Etoile** | **Link Outage Frequency, Height Perturbation, Etoile** |

| ![Mean Excess Delay Std](figures/tx_60.6_149.6_med_std.png) | ![Delay Spread Std](figures/tx_60.6_149.6_ds_std.png) |
| :---: | :---: |
| **Mean Excess Delay Standard Deviation, Height Perturbation, Etoile** | **Delay Spread Standard Deviation, Height Perturbation, Etoile** |

*Analysis of Path Gain, Mean Excess Delay, and Delay Spread Standard Deviations and Link Outage Frequency with Height Perturbation in Etoile Scene.*

## Citation
If you use OpenGERT in your research, please cite the following:

```bibtex
@misc{tadik2025opengertopensourceautomated,
      title={OpenGERT: Open Source Automated Geometry Extraction with Geometric and Electromagnetic Sensitivity Analyses for Ray-Tracing Propagation Models}, 
      author={Serhat Tadik and Rajib Bhattacharjea and Johnathan Corgan and David Johnson and Jacobus Van der Merwe and Gregory D. Durgin},
      year={2025},
      eprint={2501.06945},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2501.06945}, 
}
```

## Credits
The [NVIDIA Sionna](https://github.com/NVlabs/sionna) is Apache-2.0 licensed, as found in the [LICENSE](https://github.com/NVlabs/sionna/blob/main/LICENSE) file.

The [Microsoft GlobalMLBuildingFootprints repository](https://github.com/microsoft/GlobalMLBuildingFootprints) is licensed under the Open Data Commons Open Database License (ODbL), as detailed in the repository's [LICENSE file](https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/LICENSE).

The Digital Elevation Maps (DEMs) used in the second workflow, which involves manual building and terrain mesh creation, are sourced from the [United States Geological Survey (USGS)](https://www.usgs.gov/3d-elevation-program) website. Specifically, the 1-meter DEM storage links were manually collected from the [USGS Downloader](https://apps.nationalmap.gov/downloader/).
