from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from data_utils import NUM_CLASSES, load_datasets

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")
PLOT_PATH = Path("outputs/training_history.png")
EPOCHS = 30
BATCH_SIZE = 128


def conv_block(filters, name):
    """Two convolutions with batch normalization, followed by pooling."""
    return [
        tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu", name=name),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.25),
    ]


def build_model():
    """Build a CNN for handwritten letter classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.RandomRotation(0.05, fill_mode="constant"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
            tf.keras.layers.RandomZoom(0.1, fill_mode="constant"),
            *conv_block(32, "conv_block1"),
            *conv_block(64, "conv_block2"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu", name="dense_features"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output"),
        ],
        name="handwritten_letter_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=3, name="top3_accuracy"
            ),
        ],
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

    train_dataset, validation_dataset, _, info = load_datasets(
        batch_size=BATCH_SIZE
    )

    print("Dataset:", info.name)
    print("Training the CNN from scratch...")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_training_plot(history)

    best_epoch = history.history["val_loss"].index(min(history.history["val_loss"]))
    print(f"\nBest epoch: {best_epoch + 1}")
    print(f"Validation accuracy: {history.history['val_accuracy'][best_epoch]:.2%}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training plot saved to: {PLOT_PATH}")
    print("Run src/evaluate_model.py to measure performance on the test set.")


if __name__ == "__main__":
    main()
