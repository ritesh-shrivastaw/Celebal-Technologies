# cifar10_classifier.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Class Names ───────────────────────────────────────────────
CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# ── Load CIFAR-10 (auto-downloads ~160MB first time) ──────────
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Train images :", X_train.shape)   # (50000, 32, 32, 3)
print("Test images  :", X_test.shape)    # (10000, 32, 32, 3)
print("Train labels :", y_train.shape)