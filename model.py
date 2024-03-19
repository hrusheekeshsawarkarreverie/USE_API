# api_model.py

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
import joblib
import os
class IntentClassifierAPI:
    def __init__(self, use_model_dir="Saved_USE_model", classifier_model_path="Saved_Classifier"):
        self.use_model_dir = use_model_dir
        self.embeddings = None
        if tf.config.list_physical_devices('GPU'):
            print("GPU available, using GPU for TensorFlow computation.")
            tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
        else:
            print("No GPU available, using CPU for TensorFlow computation.")
            tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')
        self.load_models()


    def load_models(self):
        self.embeddings = tf.saved_model.load(self.use_model_dir)
        print("Loaded saved USE model")

    def run_classifier(self,sentence_list):
         sentence_list_encoded = self.embeddings(sentence_list).numpy()
         return sentence_list_encoded