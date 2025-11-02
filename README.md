# Computer Vision Lab Collection

This repository contains a collection of computer vision lab exercises for CST.620.100 Computer Vision course.

## Week 1 Lab 1

### Geometric Transformation (`src/W1.L1.geometric_transformation.py`)
Demonstrates various geometric transformations on images including:
- Image rotation
- Affine transformation

### Image Processing (`src/W1.L1.image_processing.py`)
Implements basic image processing techniques including:
- Gaussian blur with different kernel sizes
- Edge detection using Canny and Sobel methods

## Project Structure
```
├── src/
│   ├── W1.L1.geometric_transformation.py
│   └── W1.L1.image_processing.py
├── shareImage/
│   └── cat.jpg
├── README.md
└── requirements.txt
```

## Dependencies
The project requires the following Python packages:
- OpenCV (cv2)
- NumPy
- Matplotlib

See `requirements.txt` for specific version requirements.

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run individual lab files using Python:
```bash
python src/W1.L1.geometric_transformation.py
python src/W1.L1.image_processing.py
```

Each script will load and process the sample image from the `shareImage` directory and display the results using matplotlib.