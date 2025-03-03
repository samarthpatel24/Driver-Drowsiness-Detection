import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model(r'C:\Samarth\SEM 5\DL\Project\modibilenetv2_model.h5')  # Update with your model's path
class_names = ["not drowsy", "drowsy"]  # Class names based on model training order

# Define a function to make predictions
def predict_drowsiness(image):
    # Resize and normalize the input image
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_array = np.expand_dims(image_normalized, axis=0)  # Shape to (1, 224, 224, 3)

    # Make a prediction using the model
    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name

# Set up the Gradio interface
interface = gr.Interface(
    fn=predict_drowsiness,              # Function to call for predictions
    inputs=gr.Image(type="numpy"),       # Input type: image as numpy array
    outputs="text",                      # Output type: text for class name
    title="Drowsiness Detection",        # Title of the interface
    description="Upload an image to check if the person is drowsy or not."  # Description
)

# Launch the Gradio app
interface.launch()
