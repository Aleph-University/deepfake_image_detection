# Deepfake Image Detector

## Overview
Deepfake image detector is a machine learning-based application that uses EfficientNetB0 to classify images as real or fake. It leverages TensorFlow and Keras to train, evaluate, and make predictions on image datasets.

## Features
- Loads and preprocesses image datasets.
- Builds an EfficientNetB0-based deepfake detection model.
- Trains the model with adjustable hyperparameters.
- Evaluates performance using classification reports and confusion matrices.
- Saves and loads trained models for future inference.
- Predicts if an image is real or fake.

## File Structure
📂 Deepfake-Detector/

│── 📄 main.py                        # Entry point script

│── 📄 deepfake_image_detect.py       # Core model and data processing module

│── 📂 datasets/                      # Directory for test images

│── 📄 deepfake_detector.h5           # Directory for saved models

│── 📄 Figure_1.jpg                   # Training plots

│── 📄 README.md                      # Documentation

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Deepfake-Detector.git
cd Deepfake-Detector
pip install -r requirements.txt
```

## Usage
### **Training the Model**
Run `main.py` to train the model:

```bash
python main.py
```

### **Making Predictions**
After training, you can use the trained model to predict whether an image is real or fake:

```python
from deepfake_image_detect import DeepfakeDetector

detector = DeepfakeDetector(train_dir="path/to/train", val_dir="path/to/val", test_dir="path/to/test")
detector.load_model("deepfake_detector.h5")
result = detector.predict("path_to_image.jpg")
print(f"The image is predicted to be: {result}")
```

## Dependencies
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

To install all dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Contributing
Feel free to contribute by forking the repository, making improvements, and submitting a pull request.

## License
This project is licensed under the MIT License.
