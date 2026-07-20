# Img2GPS: Visual Geolocation on Penn's Engineering Quad

Img2GPS is a computer vision system that estimates where an image was captured within the University of Pennsylvania's Engineering Quad. Given a single image, the model predicts a latitude-longitude coordinate by regressing its location in a local UTM coordinate system. It contains code for:
- data collection and preprocessing;
- exploratory data analysis;
- coordinate representation;
- model architecture and training;
- evaluation;
- the interactive Gradio interface.

---

## Key Features

- **EfficientNet-B0 backbone:** Uses an ImageNet-pretrained EfficientNet-B0 as a lightweight visual feature extractor suitable for limited-data training and efficient inference.
- **UTM coordinate regression:** Converts latitude and longitude into local UTM Zone 18N coordinates, allowing the model to optimize directly in a locally Euclidean space measured in meters.
- **Custom regression head:** Replaces the classification layer with a multilayer regression head containing normalization, GELU activations, dropout, and a residual block.
- **AdamW-to-SGD fine-tuning:** Explores a two-stage optimization schedule inspired by SWATS, using AdamW for initial convergence and SGD with Nesterov momentum for later fine-tuning.
- **Robust augmentation and TTA:** Uses geometric, photometric, weather, blur, and test-time augmentations to improve robustness beyond the sunny conditions represented in the collected data.
- **Gradio application:** Provides an interactive interface for uploading an Engineering Quad image and visualizing the predicted location.

---

## Dataset

I collected and curated **935 high-resolution geotagged images** across Penn's Engineering Quad.

### Data-Collection Protocol

- **Perspective diversity:** Eight photographs were captured at approximately 45-degree intervals from each collection point.
- **Geographic labels:** Latitude and longitude were extracted automatically from smartphone EXIF metadata.
- **Capture format:** Images were captured in portrait orientation without digital zoom.
- **Geographic scope:** Collection was limited to the area bounded approximately by 33rd-34th Streets and Walnut-Spruce Streets in Philadelphia, PA.

| Split | Images |
|---|---:|
| Train | 700 |
| Validation | 118 |
| Test | 117 |
| **Total** | **935** |

The dataset is available on Hugging Face: [`EmilK5/penn-engineering-geo-location`](https://huggingface.co/datasets/EmilK5/penn-engineering-geo-location)

> NOTE: EXIF-derived GPS coordinates are approximate and may contain smartphone localization noise.

---

## Model Architecture

The model uses an EfficientNet-B0 backbone pretrained on ImageNet. Its original classification layer is replaced with a regression head:

$$
\text{Dropout}(0.3)
\rightarrow
\text{FC}(1280,512)
\rightarrow
\text{BatchNorm}
\rightarrow
\text{GELU}
\rightarrow
\text{ResidualBlock}(512)
\rightarrow
\text{FC}(512,128)
\rightarrow
\text{GELU}
\rightarrow
\text{Dropout}(0.1)
\rightarrow
\text{FC}(128,2)
$$

The two outputs represent local easting and northing offsets in UTM Zone 18N.

### Why UTM Coordinates?

Raw latitude and longitude are angular coordinates whose numerical differences do not correspond uniformly to physical distance. At the scale of Penn's campus, UTM coordinates provide a locally Euclidean representation in which prediction errors are directly measured in meters.

The model is trained using `SmoothL1Loss` on local UTM offsets. Haversine distance is used as the final geographic evaluation metric.

---

## Training

### AdamW Phase
- Epochs: 20
- Backbone learning rate: `3e-5`
- Regression-head learning rate: `3e-4`
- Scheduler: `CosineAnnealingLR`
- Weight decay: `2.4e-5`

### SGD Fine-Tuning Phase
- Epochs: 10
- Backbone learning rate: `1e-4`
- Regression-head learning rate: `1e-3`
- Momentum: `0.9`
- Nesterov momentum: enabled
- Weight decay: `1e-4`

### Augmentations

Training augmentations include:
- random resized crops;
- translation, scaling, rotation, and perspective transformations;
- brightness, contrast, gamma, and color variation;
- synthetic rain and fog;
- Gaussian and motion blur;
- occasional grayscale conversion.

At inference, test-time augmentation averages predictions from multiple transformed versions of the same image.

---

## Results

| Configuration | Mean Validation Distance |
|---|---:|
| ResNet-18 baseline | 88.37 m |
| EfficientNet-B0, AdamW | 34.38 m |
| EfficientNet-B0, AdamW -> SGD | **29.63 m** |

---

## Running the Gradio App Locally

### 1. Clone the Repository

Replace the URL below with the URL of this personal repository:

```bash
git clone https://github.com/EmilK5/Penn-IMG2GPS
cd Penn-IMG2GPS
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download or Configure the Model Checkpoint

Place the trained checkpoint at:

```text
checkpoints/efficientnet_b0_best.pt
```

Alternatively, update the checkpoint path in the application configuration.

### 5. Launch the Application

```bash
python app.py
```

Open the local Gradio URL shown in the terminal, typically:

```text
http://127.0.0.1:7860
```

---

## Limitations

- The dataset covers a small geographic region.
- Most images were collected under sunny daytime conditions.
- Smartphone EXIF coordinates contain localization noise.
- Images from nearby viewpoints may contain highly similar landmarks.
- Performance may degrade under nighttime conditions, severe weather, occlusion, or major changes to the physical environment.
- The system is intended as a campus-scale computer vision demonstration, not as a general-purpose geolocation model.

---

## Attribution

The original Img2GPS course project was developed by Grace Chi, Darren Weng, and Emil Kielar for CIS 5190: Applied Machine Learning at the University of Pennsylvania.

This repository contains my personal reimplementation and extensions. The original team repository is available [here](https://github.com/EmilK5/CIS5190-final-project).
