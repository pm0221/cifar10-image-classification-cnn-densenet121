# CIFAR-10 Image Classification Using CNN and DenseNet121

## Project Overview
This project builds deep learning models to classify small colour images into 10 object categories using the CIFAR-10 dataset. It explores image visualization, preprocessing, data augmentation, convolutional neural networks (CNNs), transfer learning with DenseNet121, and model evaluation.

The implementation is provided in the Jupyter Notebook `cv5.ipynb`.

## Objectives
- Load and explore the CIFAR-10 image dataset.
- Visualize sample images and class distributions.
- Normalize image pixel values and one-hot encode class labels.
- Build and train a CNN for image classification.
- Apply image augmentation during training.
- Evaluate model performance using accuracy, precision, recall, confusion matrix, and classification report.
- Visualize individual predictions and their class probabilities.
- Experiment with DenseNet121 transfer learning.
- Save the trained model and classify an external image.

## Dataset
The project uses the **CIFAR-10** dataset provided by TensorFlow/Keras. It contains 60,000 colour images of size 32 × 32 pixels across 10 classes.

The classes are:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is loaded automatically using:

```python
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

The standard dataset split contains 50,000 training images and 10,000 test images.

## Project Workflow

### 1. Import Libraries
The notebook uses TensorFlow/Keras, NumPy, Pandas, Matplotlib, and Scikit-learn metrics for model development, data handling, visualization, and evaluation.

### 2. Dataset Exploration
- Load the CIFAR-10 training and testing sets.
- Inspect the shapes of the image and label arrays.
- Display a grid of sample training images with their class names.
- Plot the class distribution for both training and testing sets.

### 3. Data Preprocessing
- Normalize image pixel values by dividing by 255, scaling them to the range 0–1.
- Convert integer class labels into one-hot encoded vectors for the 10 classes.
- Use images with input dimensions of 32 × 32 × 3.

### 4. CNN Model Architecture
The first model is a Sequential CNN with:
- Convolutional layers using 32, 64, and 128 filters.
- 3 × 3 convolution kernels and ReLU activation.
- Batch normalization.
- Max pooling layers for spatial downsampling.
- Dropout layers to help reduce overfitting.
- A Flatten layer.
- A Dense layer with 128 units and ReLU activation.
- A final Dense layer with 10 units and softmax activation.

The model is compiled with categorical cross-entropy loss, the Adam optimizer, and accuracy, precision, and recall metrics.

### 5. Data Augmentation and CNN Training
The notebook applies image augmentation with:
- Width shift range: 0.1
- Height shift range: 0.1
- Horizontal flipping

The CNN is trained with a batch size of 32 for up to 50 epochs. The test dataset is supplied as validation data during training.

An EarlyStopping callback is defined with validation loss monitoring and patience of 2, but it is not enabled in the shown training call.

### 6. Training Performance Visualization
The notebook plots training and validation curves for:
- Loss
- Accuracy
- Precision
- Recall

### 7. Model Evaluation
The CNN is evaluated on the test set. The notebook generates:
- Test accuracy
- Confusion matrix
- Classification report with class-wise evaluation metrics

### 8. Prediction Visualization
The notebook displays sample test images and predicted classes. It also compares predicted class probabilities with the true labels, highlighting correct and incorrect predictions.

### 9. DenseNet121 Transfer Learning
A second model is built using DenseNet121 with ImageNet weights:
- Input shape: 32 × 32 × 3
- `include_top=False`
- Global average pooling
- A final Dense layer with 10 softmax outputs

This model is compiled with categorical cross-entropy loss and Adam optimizer, then trained using the same augmented training generator for up to 100 epochs.

### 10. Save the Model
The notebook saves the model to:

```text
cnn_20_epochs.h5
```

### 11. Predict an External Image
The final notebook cell loads an external image named `airplan.jpg`, resizes it to 32 × 32 pixels, normalizes it, and predicts its CIFAR-10 class and confidence.

Make sure `airplan.jpg` is available in the notebook's working directory before running this final cell.

## Technologies and Libraries
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Installation

Install the required packages:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn jupyter
```

## How to Run
1. Open the project in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the required libraries.
3. Open `cv5.ipynb`.
4. Run the notebook cells in order. The CIFAR-10 dataset will be downloaded automatically by Keras when first loaded.
5. To run the final external-image prediction cell, place `airplan.jpg` in the working directory.
6. Review the training curves, evaluation results, confusion matrix, and predictions.

## Project Structure

```text
CIFAR10_Image_Classification/
├── cv5.ipynb
├── README.md
├── airplan.jpg          # Required for the external-image prediction cell
└── cnn_20_epochs.h5     # Generated when the model-saving cell runs
```

## Notes
- The notebook first trains a custom CNN and later replaces the `model` variable with a DenseNet121-based model.
- The model saved in `cnn_20_epochs.h5` is whichever model is assigned to `model` when the save cell is executed. In the notebook's current order, this is the DenseNet121 model, provided the preceding cells run successfully.
- The test set is also used as validation data during training in the notebook.
- The final external-image prediction cell requires `airplan.jpg`; this image is not part of the CIFAR-10 dataset download.
- Training duration and evaluation results depend on the runtime environment and training execution.

## Author
Prayag Maru
