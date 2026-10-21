
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_utils import NUM_CLASSES, load_datasets

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")


def main():
    assert MODEL_PATH.exists(), (
        f"Model file not found: {MODEL_PATH}. Run train.py first."
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    _, _, test_dataset, _ = load_datasets()

    images, labels = next(iter(test_dataset))
    probabilities = model.predict(images, verbose=0)

    assert probabilities.shape == (images.shape[0], NUM_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

    predicted_labels = np.argmax(probabilities, axis=1)

    print("Model loaded successfully.")
    print("Prediction shape:", probabilities.shape)
    print("First actual label:", chr(ord("A") + int(labels[0].numpy())))
    print(
        "First predicted letter:",
        chr(ord("A") + int(predicted_labels[0])),
    )
    print(
        "First prediction confidence:",
        f"{probabilities[0, predicted_labels[0]]:.2%}",
    )
    print("All model checks passed!")


if __name__ == "__main__":
    main()
