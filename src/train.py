
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import NUM_CLASSES, load_datasets

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")
PLOT_PATH = Path("outputs/training_history.png")
EPOCHS = 15
BATCH_SIZE = 128


def build_model():
    """Build a CNN for handwritten letter classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="handwritten_letter_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_plot(history):
    """Save training and validation accuracy and loss curves."""
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=150)
    plt.close(figure)


def main():
    tf.keras.utils.set_random_seed(42)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, test_dataset, info = load_datasets(
        batch_size=BATCH_SIZE
    )

    print("Dataset:", info.name)
    print("Training the CNN from scratch...")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    save_training_plot(history)

    print("\nEvaluating the model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
