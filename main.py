# main.py

from deepfake_image_detect import DeepfakeDetector

# Define dataset directories
dir = "Example Directory"
train_dir = dir + '/Train'
val_dir = dir + '/Validation'
test_dir = dir + '/Test'

# Initialize the detector with dataset directories
detector = DeepfakeDetector(train_dir, val_dir, test_dir)

# Load the data
detector.load_data()

# Build the model
detector.build_model(learning_rate=0.001)

# Train the model
history = detector.train_model(epochs=2)

# Evaluate the model on test set
detector.evaluate_model()

# Generate classification report and confusion matrix
detector.generate_classification_report()

# Plot training history
detector.plot_training_history(history)

# Save the trained model
detector.save_model('deepfake_detector.h5')

# Load the model for future use
detector.load_model('deepfake_detector.h5')

# Predict on a new image
result = detector.predict('path_to_image.jpg')
print(f'The image is predicted to be: {result}')
