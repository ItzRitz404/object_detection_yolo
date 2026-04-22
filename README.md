# Object Detection with YOLO

A Python-based object detection project built with **Ultralytics YOLO**.

This repository includes scripts for:

- Converting annotation files into YOLO label format
- Generating a `data.yaml` file automatically
- Training a custom YOLO model
- Running inference on test images
- Serving predictions through a FastAPI API

---

## Project Overview

This project follows a custom object detection workflow:

1. Raw annotation JSON files are converted into YOLO label files
2. Class names are collected automatically
3. A YOLO dataset config (`data.yaml`) is generated
4. A YOLO model is trained using Ultralytics
5. Predictions can be tested locally or exposed through an API

---

## Repository Structure

```
object_detection_yolo/
├── check.py                 # Check CUDA / GPU availability
├── createYaml.py            # Generate data.yaml from classes.txt
├── generateLabels.py        # Convert JSON annotations to YOLO labels
├── train.py                 # Train or resume YOLO training
├── tester.py                # Run inference on local images
├── yolo.py                  # FastAPI detection API
├── image64.txt              # Example base64 image payload
├── images/                  # Sample images for testing
└── runs/detect/train2/      # Saved training results and weights
```

---

## Features

- Convert JSON annotations into YOLO label format
- Automatically generate `classes.txt`
- Automatically generate `data.yaml`
- Train a YOLO model using Ultralytics
- Test predictions on local images
- Expose object detection through a FastAPI endpoint
- Store training outputs such as weights and result plots

---

## Requirements

Install the required dependencies:

```bash
pip install ultralytics torch fastapi uvicorn pillow numpy pydantic
```

> If you want GPU acceleration, install a CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Clone the Repository

```bash
git clone https://github.com/ItzRitz404/object_detection_yolo.git
cd object_detection_yolo
```

---

## Important Note About Paths

Some scripts currently use absolute local file paths, for example:

```
<PROJECT_ROOT>/object_detection/
```

Before running this project on another machine, update those paths inside:

- `generateLabels.py`
- `createYaml.py`
- `train.py`
- `tester.py`

> A good improvement would be replacing hard-coded paths with relative paths or environment variables (see [Suggested Improvements](#suggested-improvements)).

---

## Dataset Preparation

### 1. Generate YOLO Labels

```bash
python generateLabels.py
```

This script:

- Reads annotation JSON files
- Extracts all class labels automatically
- Creates a class-to-index mapping
- Converts bounding boxes into YOLO format
- Writes label files into training and validation folders

Expected output:

```
dataset/labels/train/
dataset/labels/val/
dataset/classes.txt
```

### 2. Generate `data.yaml`

```bash
python createYaml.py
```

This creates:

```
dataset/data.yaml
```

Example structure:

```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: class_name_1
  1: class_name_2
```

---

## Check GPU Availability

To verify whether PyTorch detects your GPU:

```bash
python check.py
```

---

## Training

To start or resume training:

```bash
python train.py
```

The training script uses Ultralytics YOLO and can resume from an existing checkpoint.

Training outputs are saved under:

```
runs/detect/train2/
```

This folder includes:

- `best.pt` — best model weights
- `last.pt` — most recent checkpoint
- Confusion matrix
- Result plots
- Training summaries

---

## Local Inference

To run inference on local images:

```bash
python tester.py
```

This loads your trained model and runs predictions on images from a local folder. Place test images inside the `images/` directory and adjust the source path in `tester.py` if needed.

---

## API Inference with FastAPI

The repository includes a FastAPI app for object detection in `yolo.py`.

### Start the API

```bash
uvicorn yolo:app --reload
```

### Endpoint

```
POST /detect
```

### Request Body

The API expects a JSON payload containing a base64-encoded image:

```json
{
  "img": "data:image/jpeg;base64,..."
}
```

### Example Python Request

```python
import requests
import base64

with open("images/sign.jpeg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "img": f"data:image/jpeg;base64,{encoded}"
}

response = requests.post("http://127.0.0.1:8000/detect", json=payload)
print(response.json())
```

### Example Response

```json
{
  "detections": [
    {
      "name": "object_name",
      "confidence": 0.98,
      "box": {
        "x1": 120.4,
        "y1": 80.2,
        "x2": 412.7,
        "y2": 395.1
      }
    }
  ]
}
```

---

## Sample Files Included

This repository includes:

- Sample images in `images/`
- An example base64 payload in `image64.txt`
- Previous training outputs in `runs/detect/train2/`

These can help with quick testing and validation.

---

## Suggested Improvements

Possible next steps for the project:

- Add a `requirements.txt`
- Replace absolute paths with configurable relative paths
- Add a `.env` or config file for environment-specific settings
- Document the dataset source and full class list
- Add prediction screenshots to the README
- Include evaluation metrics and model performance summary

---

## License

No license file is currently included in this repository.

If you plan to share or reuse this project publicly, consider adding a license such as [MIT](https://choosealicense.com/licenses/mit/).

---

## Author

Created by [ItzRitz404](https://github.com/ItzRitz404)
