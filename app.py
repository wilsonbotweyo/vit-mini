from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define the class labels
class_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
input_shape = (32, 32, 3)  # Update with the appropriate input shape of your model
num_classes = 10  # Update with the number of classes in your model

# Define a small Vision Transformer model
def create_vit_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Resize images to a smaller size to reduce computation
    resized_inputs = preprocessing.Rescaling(scale=0.25)(inputs)

    # Patch embedding layer
    patches = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(resized_inputs)
    patches = layers.BatchNormalization()(patches)
    patches = layers.Activation("swish")(patches)

    # Flattening patches
    patches = layers.Flatten()(patches)

    # MLP with two hidden layers
    mlp = layers.Dense(256, activation="swish")(patches)
    mlp = layers.Dropout(0.5)(mlp)
    mlp = layers.Dense(128, activation="swish")(mlp)
    mlp = layers.Dropout(0.5)(mlp)
    # Classification head
    outputs = layers.Dense(num_classes, activation="softmax")(mlp)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load the trained model
model = create_vit_model(input_shape, num_classes)
model.load_weights("vit_model_weights.h5")  # Replace with the path to your trained model weights

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']
        
        # Preprocess the uploaded image
        image = Image.open(image_file).convert("RGB")
        image = image.resize((32, 32))  # Resize the image to match the input shape of the model
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions using the trained model
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        
        # Render the template with the predicted label
        return render_template('index.html', label=predicted_label)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
