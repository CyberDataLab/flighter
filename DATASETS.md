## Overview of Datasets

In our study, we employed three distinct datasets to evaluate the performance and robustness of the Flighter model across various scenarios: the MSTAR dataset, the SAMPLE dataset, and the OpenSARShip dataset. Each dataset offers unique characteristics and challenges, contributing to a comprehensive assessment of the model's capabilities.

---

## 1. MSTAR Dataset

### Description
The Moving and Stationary Target Acquisition and Recognition (MSTAR) dataset is a collection of Synthetic Aperture Radar (SAR) images developed by the Air Force Research Laboratory (AFRL) and the Defense Advanced Research Projects Agency (DARPA). It serves as a benchmark for automatic target recognition in SAR imagery.

### Key Features
- **Content**: SAR images of various military vehicles, including tanks, armored personnel carriers, and air defense units.
- **Resolution**: X-band SAR imagery with a resolution of one foot.
- **Collection Details**: Data collected in 1995 at Redstone Arsenal, Huntsville, AL, by Sandia National Laboratory's SAR sensor platform.
- **Target Types**: Includes T-72 tanks, BMP2 infantry fighting vehicles, BTR-70 armored personnel carriers, and a "Slicy" geometric-shaped target.

### Reference
For detailed information and access to the dataset, refer to the official MSTAR overview provided by the AFRL and DARPA: [MSTAR Dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar)

---

## 2. SAMPLE Dataset

### Description
The Synthetic and Measured Paired Labeled Experiment (SAMPLE) dataset is a collection of SAR imagery that pairs measured data from the MSTAR collection with simulated SAR imagery. It is designed to facilitate the development and evaluation of algorithms for SAR image analysis.

### Key Features
- **Content**: Paired measured and synthetic SAR images for ten public MSTAR target classes.
- **Purpose**: Enhances training and evaluation diversity by providing both real and simulated data.
- **Availability**: Publicly released version includes data with azimuth angles between 10 and 80 degrees.

### Reference
The SAMPLE dataset is available on the following publication:

> Benjamin Lewis et al., "A SAR dataset for ATR development: the Synthetic and Measured Paired Labeled Experiment (SAMPLE)," in Algorithms for Synthetic Aperture Radar Imagery XXVI, May. 2019, doi: 10.1117/12.2523460

---

## 3. OpenSARShip Dataset

### Description
The OpenSARShip dataset is dedicated to the interpretation of ship targets in Sentinel-1 SAR imagery. It provides a comprehensive collection of SAR ship chips integrated with Automatic Identification System (AIS) messages, facilitating research in maritime surveillance and ship classification.

### Key Features
- **Content**: 11,346 SAR ship chips extracted from 41 Sentinel-1 images.
- **Ship Types**: Includes various ship types such as tankers, container ships, and bulk carriers.
- **Data Source**: Sentinel-1 C-band SAR imagery.
- **Properties**: The dataset is characterized by specificity, large scale, diversity, reliability, and public availability.

### Reference
For more information and access to the dataset, refer to the following publication:

> L. Huang et al., "OpenSARShip: A Dataset Dedicated to Sentinel-1 Ship Interpretation," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 11, no. 1, pp. 195-208, Jan. 2018, doi: 10.1109/JSTARS.2017.2755672.
