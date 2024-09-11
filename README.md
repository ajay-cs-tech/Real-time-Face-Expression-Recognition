# Real-time Face Expression Recognition

This project implements a real-time face expression recognition system using Convolutional Neural Networks (CNN) and OpenCV. The model is trained to recognize seven different facial expressions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Face detection using Haar Cascade Classifier
- Custom CNN model for expression recognition
- Real-time video capture and processing
- Data augmentation for improved model performance
- Pre-trained model saving and loading

## Requirements

- Python 3.11 (latest stable version as of September 2024)
- OpenCV
- TensorFlow
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ajay-cs-tech/Real-time Face Expression Recognition.git
   cd face-expression-recognition
   ```

2. Install the required packages:
   ```
   pip install opencv-python tensorflow numpy
   ```

3. Download the Haar Cascade file for face detection (if not already included):
   ```
   wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
   ```

## Usage

1. Prepare your dataset:
   - Organize your training data in the following structure:
     ```
     D:/ml_projects/data/train/
     ├── Angry
     ├── Disgust
     ├── Fear
     ├── Happy
     ├── Sad
     ├── Surprise
     └── Neutral
     ```
   - Each subdirectory should contain the respective expression images.

2. Train the model:
   - Run the script:
     ```
     python face_recog.py
     ```
   - This will train the model and save it as 'expression_model.h5'

3. Real-time detection:
   - After training, the script will automatically start real-time detection using your computer's webcam.
   - Press 'q' to quit the application.

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers with MaxPooling
- Flatten layer
- Dense layer with Dropout
- Output layer with 7 units (one for each expression)

## Future Improvements

- Implement transfer learning using pre-trained models like VGG or ResNet
- Add support for multiple face detection and recognition
- Improve model accuracy with more data and fine-tuning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
