# OpenGERT: Open Source Automated Geometry Extraction with Geometric and Electromagnetic Sensitivity Analyses for Ray-Tracing Propagation Models

| ![Seattle](figures/seattle3.png) | ![Georgia Tech - Atlanta](figures/georgia_tech.png) | ![Manhattan](figures/manhattan.png) |
| :---: | :---: | :---: |
| **Seattle** | **Georgia Tech - Atlanta** | **Manhattan** |

*Seattle, Georgia Tech, and Manhattan represented in Sionna RT through automatic geometry extraction pipeline.*

## Workflow

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