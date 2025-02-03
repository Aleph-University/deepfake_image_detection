# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

class DeepfakeDetector:
    def __init__(self, train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the DeepfakeDetector with dataset directories and model parameters.
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None

    def load_data(self):
        """
        Load the training, validation, and test datasets.
        """
        datagen = ImageDataGenerator(rescale=1.0/255.0)

        self.train_gen = datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        self.val_gen = datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        self.test_gen = datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

    def build_model(self, learning_rate=0.001):
        """
        Build the deepfake detection model using EfficientNetB0.
        """
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=10):
        """
        Train the deepfake detection model.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('deepfake_detector.h5', save_best_only=True, monitor='val_loss')

        history = self.model.fit(
            self.train_gen,
            epochs=epochs,
            validation_data=self.val_gen,
            callbacks=[early_stopping, model_checkpoint]
        )

        return history

    def evaluate_model(self):
        """
        Evaluate the model on the test dataset.
        """
        test_loss, test_accuracy = self.model.evaluate(self.test_gen)
        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')

    def generate_classification_report(self):
        """
        Generate a classification report and confusion matrix.
        """
        predictions = self.model.predict(self.test_gen)
        predicted_classes = (predictions > 0.5).astype(int)
        true_classes = self.test_gen.classes
        class_labels = list(self.test_gen.class_indices.keys())

        print("Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))

        print("Confusion Matrix:")
        print(confusion_matrix(true_classes, predicted_classes))

    def plot_training_history(self, history):
        """
        Plot training history including accuracy and loss.
        """
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.legend()

        plt.show()

    def save_model(self, file_path='deepfake_detector.h5'):
        """
        Save the trained model.
        """
        self.model.save(file_path)

    def load_model(self, file_path='deepfake_detector.h5'):
        """
        Load a pre-trained model.
        """
        self.model = tf.keras.models.load_model(file_path)

    def predict(self, img_path):
        """
        Predict if an image is real or fake.
        """
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        return 'Fake' if prediction[0][0] > 0.5 else 'Real'
