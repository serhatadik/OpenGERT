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

   - **Blosm 2.7.8:**
     - **Download Link:** [Blosm 2.7.8 on Gumroad](https://prochitecture.gumroad.com/l/blender-osm)

   - **Mitsuba-Blender Add-on (Version 0.3.0 or Nightly Release):**
     - **Download Link:** [Mitsuba-Blender Releases](https://github.com/mitsuba-renderer/mitsuba-blender/releases)
     - **Important:** *Do **not** use Mitsuba-Blender v0.4.0, as it is **not** compatible and will not work.*

### **Function 2: Sensitivity Analysis Using Pre-Extracted Scenes**

If you already have the scene `.xml` file and meshes ready, or prefer to use the pre-existing Munich or Etoile scenes in Sionna RT without performing geometry extraction, follow these steps:

1. **Clone the OpenGERT Repository**

   Open your terminal or command prompt and execute the following command:
   ```
   git clone https://github.com/serhatadik/OpenGERT.git
   cd OpenGERT
   ```
2. **Set Up a Virtual Environment and Install OpenGERT**

   - **Create a Virtual Environment:**
     ```
     python3 -m venv venv
     ```
     *This command creates a virtual environment named `venv`.*

   - **Activate the Virtual Environment:**

     - **On macOS/Linux:**
       ```
       source venv/bin/activate
       ```
     - **On Windows:**
       ```
       venv\Scripts\activate
       ```
   - **Install the OpenGERT Package:**
     ```
     pip install -e .
     ```
     *This command installs the OpenGERT package in editable mode.*

3. **Verify the Installation**

   - Ensure that the OpenGERT package is installed correctly by running:
     ```
     pip list
     ```
     *You should see `OpenGERT` listed among the installed packages.*

4. **Using Pre-Extracted Scenes**

   - **Available Scenes:** Munich and Etoile
   - **Usage Instructions:**
     - Refer to the [OpenGERT Documentation](https://github.com/yourusername/OpenGERT#readme) for detailed instructions on how to load and utilize the pre-extracted scenes within Sionna RT.
     - Ensure that the scene `.xml` files and associated meshes are placed in the correct directories as specified in the documentation.

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

## Credits
The [NVIDIA Sionna](https://github.com/NVlabs/sionna) is Apache-2.0 licensed, as found in the [LICENSE](https://github.com/NVlabs/sionna/blob/main/LICENSE) file.

The [Microsoft GlobalMLBuildingFootprints repository](https://github.com/microsoft/GlobalMLBuildingFootprints) is licensed under the Open Data Commons Open Database License (ODbL), as detailed in the repository's [LICENSE file](https://github.com/microsoft/GlobalMLBuildingFootprints/blob/main/LICENSE).

The Digital Elevation Maps (DEMs) used in the second workflow, which involves manual building and terrain mesh creation, are sourced from the [United States Geological Survey (USGS)](https://www.usgs.gov/3d-elevation-program) website. Specifically, the 1-meter DEM storage links were manually collected from the [USGS Downloader](https://apps.nationalmap.gov/downloader/).