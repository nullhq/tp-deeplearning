from tensorflow import keras
import mlflow

EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)

    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Précision sur les données de test: {test_acc:.4f}")

    mlflow.log_metric("test_accuracy", test_acc)

    model.save("mnist_model.h5")
    print("Modèle sauvegardé sous mnist_model.h5")

    mlflow.keras.log_model(model, "mnist-model")