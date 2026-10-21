
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from data_utils import NUM_CLASSES, load_datasets

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")
OUTPUT_DIR = Path("outputs")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_dataset, _ = load_datasets()

    print("Generating predictions on the test dataset...")
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    true_labels = np.concatenate(
        [labels.numpy() for _, labels in test_dataset]
    )

    label_ids = list(range(NUM_CLASSES))
    letters = [chr(ord("A") + index) for index in label_ids]

    accuracy = accuracy_score(true_labels, predicted_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_ids,
        target_names=letters,
        digits=4,
        zero_division=0,
    )

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(report)

    report_path = OUTPUT_DIR / "classification_report.txt"
    report_path.write_text(
        f"Test accuracy: {accuracy:.4%}\n\n{report}",
        encoding="utf-8",
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_ids,
    )

    figure, axis = plt.subplots(figsize=(14, 12))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=letters,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title("EMNIST Letters - Confusion Matrix")
    figure.tight_layout()

    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    figure.savefig(matrix_path, dpi=180)
    plt.close(figure)

    print(f"\nClassification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {matrix_path}")


if __name__ == "__main__":
    main()
