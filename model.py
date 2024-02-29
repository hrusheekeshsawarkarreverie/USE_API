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
        self.classifier_model_path = classifier_model_path
        self.embeddings = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.load_models()

    def load_models(self):
            self.embeddings = tf.saved_model.load(self.use_model_dir)
            print("Loaded saved USE model")

    def run_classifier(self,sentence_list,intent_list,num_intents):
        try:
            self.model = tf.keras.models.load_model(self.classifier_model_path)
        except :  # Handle the specific error when the model is not found
            sentence_list = [s.lower() for s in sentence_list]
            sentence_list = self.embeddings(sentence_list).numpy()
            intent_list = self.label_encoder.fit_transform(intent_list)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(512,), dtype=tf.float32),  # 512 is the dimension of the USE embeddings
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_intents, activation='softmax')
                # Output layer with the number of classes
            ])
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(sentence_list, intent_list, epochs=100, batch_size=8)
            os.makedirs(self.classifier_model_path, exist_ok=True)
            model.save(self.classifier_model_path)
            self.model = model


    def predict_intent(self, sentence):
        sentence = sentence.lower()
        sentence_encoded = self.embeddings([sentence])
        predictions = self.model.predict(sentence_encoded)
        predicted_label = self.label_encoder.inverse_transform([np.argmax(predictions)])
        return predicted_label[0]

# Example Usage:
# if __name__ == "__main__":
#     api = IntentClassifierAPI()
#     new_sentence = "How to file an Income Tax Return?"
#     predicted_label = api.predict_intent(new_sentence)
#     print(f"Sentence is predicted as: {predicted_label}")
