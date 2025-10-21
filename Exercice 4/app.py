from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# Chargement du modèle Keras
model = keras.models.load_model('mnist_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': "Aucun fichier n'a été fournis !"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image_data = np.array(image).reshape(1, 784).astype("float32") / 255.0

    prediction = model.predict(image_data)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    return jsonify({
        'prediction': predicted_class,
        'probabilities': prediction.tolist()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)