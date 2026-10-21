import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support,
                             top_k_accuracy_score)

from data_utils import NUM_CLASSES, load_datasets

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")
DEFAULT_OUTPUT_DIR = Path("outputs")
LABEL_IDS = list(range(NUM_CLASSES))
LETTERS = [chr(ord("A") + index) for index in LABEL_IDS]


def expected_calibration_error(probabilities, true_labels, bins=10):
    """Average gap between confidence and accuracy across confidence bins."""
    confidences = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == true_labels
    edges = np.linspace(0.0, 1.0, bins + 1)

    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if np.any(in_bin):
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            error += in_bin.mean() * gap

    return error


def top_confusions(matrix, limit=10):
    """Return the most frequent (true, predicted, count) error pairs."""
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    order = np.argsort(errors, axis=None)[::-1][:limit]
    pairs = [np.unravel_index(index, errors.shape) for index in order]
    return [(true, pred, int(errors[true, pred])) for true, pred in pairs]


def save_confusion_matrix(matrix, path):
    normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100

    figure, axis = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(
        confusion_matrix=normalized,
        display_labels=LETTERS,
    )
    display.plot(ax=axis, cmap="Blues", values_format=".0f", colorbar=False)
    axis.set_title("EMNIST Letters - Confusion Matrix (% per true letter)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_misclassified_examples(images, true_labels, predicted_labels,
                                probabilities, path, limit=24):
    wrong = np.flatnonzero(true_labels != predicted_labels)
    rng = np.random.default_rng(42)
    selected = rng.choice(wrong, size=min(limit, len(wrong)), replace=False)

    columns = 8
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 2.4))

    for axis in np.ravel(axes):
        axis.axis("off")

    for axis, index in zip(np.ravel(axes), selected):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(
            f"{LETTERS[true_labels[index]]} -> {LETTERS[predicted_labels[index]]}"
            f"\n{probabilities[index].max():.0%}",
            fontsize=9,
        )

    figure.suptitle("Misclassified examples (true -> predicted, confidence)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_dataset, _ = load_datasets()

    print("Generating predictions on the test dataset...")
    batches = [(images.numpy(), labels.numpy()) for images, labels in test_dataset]
    images = np.concatenate([batch_images for batch_images, _ in batches])
    true_labels = np.concatenate([batch_labels for _, batch_labels in batches])

    probabilities = model.predict(images, batch_size=256, verbose=1)
    predicted_labels = probabilities.argmax(axis=1)

    accuracy = accuracy_score(true_labels, predicted_labels)
    top3_accuracy = top_k_accuracy_score(
        true_labels, probabilities, k=3, labels=LABEL_IDS
    )
    loss = log_loss(true_labels, probabilities, labels=LABEL_IDS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=LABEL_IDS, average="macro",
        zero_division=0,
    )
    calibration_error = expected_calibration_error(probabilities, true_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABEL_IDS,
        target_names=LETTERS,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABEL_IDS)
    confusions = "\n".join(
        f"  {LETTERS[true]} -> {LETTERS[pred]}: {count}"
        for true, pred, count in top_confusions(matrix)
    )

    summary = (
        f"Test samples: {len(true_labels)}\n"
        f"Accuracy: {accuracy:.4%}\n"
        f"Top-3 accuracy: {top3_accuracy:.4%}\n"
        f"Log loss: {loss:.4f}\n"
        f"Macro precision: {precision:.4f}\n"
        f"Macro recall: {recall:.4f}\n"
        f"Macro F1-score: {f1:.4f}\n"
        f"Expected calibration error (ECE): {calibration_error:.4f}\n"
        f"\nMost frequent confusions (true -> predicted: count):\n{confusions}\n"
    )

    print("\n" + summary)
    print(report)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(f"{summary}\n{report}", encoding="utf-8")

    matrix_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(matrix, matrix_path)

    examples_path = output_dir / "misclassified_examples.png"
    save_misclassified_examples(
        images, true_labels, predicted_labels, probabilities, examples_path
    )

    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {matrix_path}")
    print(f"Misclassified examples saved to: {examples_path}")


if __name__ == "__main__":
    main()
