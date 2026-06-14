# CIFAR-10 Image Classification Project

This project trains and compares different image classification models on the CIFAR-10 dataset:

- ANN (Artificial Neural Network) using flattened image pixels
- CNN (Convolutional Neural Network) using spatial image features
- Additional CNN variants with data augmentation and EarlyStopping

## What this project does

1. Loads the CIFAR-10 dataset
2. Visualizes sample images
3. Trains an ANN model
4. Trains a CNN model
5. Trains an augmented CNN model
6. Compares performance using test accuracy and loss
7. Saves model outputs, comparison CSV, and plots

## Requirements

Install the required packages:

```bash
pip install tensorflow matplotlib numpy pandas
```

## Run the project

From the project folder, run:

```bash
python cifar10_classifier.py
```

## Output files

The script generates:

- ann_cifar10.h5
- cnn_cifar10.h5
- aug_cnn_cifar10.h5
- model_comparison.csv
- plot_samples.png
- plot_ann_vs_cnn_accuracy.png
- plot_ann_vs_cnn_loss.png
- plot_final_comparison.png

## Notes

- CNN usually performs better than ANN on image data because it preserves spatial structure.
- Data augmentation and EarlyStopping help improve generalization and reduce overfitting.
