# Img2GPS: Penn Engineering Quad Geolocation

A supervised computer vision pipeline designed to predict exact GPS coordinates from raw images taken within the University of Pennsylvania's engineering quad (bounded by 33rd-34th Streets and Walnut-Spruce Streets). Given a raw image, the model localizes where the photo was taken to within roughly 30 meters on average.

This repository contains the full machine learning training pipeline, the model architecture, and an interactive Gradio web application for real-time inference.

---

## 🚀 Key Features

* **EfficientNet-B0 Backbone:** Utilizes a lightweight, ImageNet-pretrained feature extractor ideal for low parameter counts and fast deployment.
* **UTM Coordinate Regression:** Converts raw latitude/longitude to Universal Transverse Mercator (UTM Zone 18N) meter-space to allow locally Euclidean, physical distance-aligned optimization.
* **SWATS Optimization:** Implements an exploratory two-phase optimization routing—starting with AdamW for rapid initial convergence and automatically transitioning to SGD with Nesterov momentum to settle into flatter, more generalizable minima.
* **Robust Data Augmentation & TTA:** Compensates for sunny-day data using synthetic rain, fog, lighting, and geometric variations, alongside Test-Time Augmentation (TTA) averaging at inference.
* **Gradio Web App:** An intuitive, user-facing interface allowing individuals to drop an image of the quad and instantaneously receive a predicted coordinate overlay.

---

## 📊 Dataset & Spatial Distribution

We collected and open-sourced a high-resolution dataset containing **2,816 images (~14.3 GB)** across the Penn engineering quad. 

### Data Collection Protocol
* **Perspective Diversity:** 8 photos taken per fixed point at $45^\circ$ intervals in portrait orientation.
* **Ground Truth:** Precision GPS labels extracted directly from raw EXIF image metadata.
* **Augmentation Split:** Supplemented by course-provided public data for a total training footprint:

| Split | Images | Source |
| :--- | :--- | :--- |
| **Train** | 5,317 | Own collection + course-provided public |
| **Validation** | 1,050 | Own collection + course-provided public |
| **Test (Hidden)** | 681 | Course-provided hidden |

> **Dataset Availability:** The full compiled dataset can be found publicly on Hugging Face at [`gracech1/combined-penn-engineering-geo-location`](https://huggingface.co/datasets/gracech1/combined-penn-engineering-geo-location).

---

## 🛠️ Model Architecture & Training

### Regression Head Pipeline
The default classification layer of EfficientNet-B0 is swapped for a deeper, gradient-stable head featuring skip connections:
$$\text{Dropout}(0.3) \rightarrow \text{FC}(1280 \rightarrow 512) \rightarrow \text{BN} + \text{GELU} \rightarrow \text{ResidualBlock}(512) \rightarrow \text{FC}(512 \rightarrow 128) \rightarrow \text{GELU} \rightarrow \text{Dropout}(0.1) \rightarrow \text{FC}(128 \rightarrow 2)$$

### Optimization Logic
* **Loss Function:** Trained via `SmoothL1Loss` ($\beta = 0.1$) on local UTM meter offsets. Standard Haversine loss was intentionally avoided due to vanishing gradients near zero ($arcsin(\sqrt{\cdot})$).
* **SWATS Training Schedule:**
    1.  **AdamW Phase:** 20 epochs (Backbone $lr=\text{3e-5}$, Head $lr=\text{3e-4}$, CosineAnnealingLR).
    2.  **SGD Phase:** 10 epochs with Nesterov momentum (Backbone $lr=\text{1e-4}$, Head $lr=\text{1e-3}$, Momentum $=0.9$).

---

## 📈 Results & Performance

| Configuration | Val Distance (m) | Test Haversine (m) ↓ |
| :--- | :--- | :--- |
| **Baseline (ResNet-18, MSE, raw lat/lon)** | 88.37m | — |
| **EfficientNet-B0 (AdamW only)** | 34.38m | — |
| **EfficientNet-B0 + SWATS (Final Model)** | **29.63m** | **60.37m** |

Our final SWATS-optimized pipeline secured **Rank 10** on the official project leaderboard.

---

## 💻 Running the Gradio App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EmilK5/CIS5190-final-project.git
   cd CIS5190-final-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application:**
   ```bash
   python app.py
   ```
   Navigate to the local URL (typically `http://127.0.0.1:7860`) generated in your terminal to start predicting locations interactively!

---

## 👥 Team Members & Contributions
* **Emil Kielar:** Data collection, exploratory data analysis, model design, model training, and report writing.
* **Grace Chi:** Data collection, web application development, model training, and report writing.
* **Darren Weng:** Data collection, report writing, and demo script creation.
