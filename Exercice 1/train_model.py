import tensorflow as tf
from tensorflow import keras
import numpy as np

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation des données
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionnement des images pour les réseaux fully-connected
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Construction du modèle
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement du modèle
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# Évaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision sur les données de test: {test_acc:.4f}")

# Sauvegarde du modèle
model.save("mnist_model.h5")
print("Modèle sauvegardé sous mnist_model.h5")
