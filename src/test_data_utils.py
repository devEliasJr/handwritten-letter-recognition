
import numpy as np

from data_utils import NUM_CLASSES, load_datasets


def main():
    train_dataset, validation_dataset, test_dataset, info = load_datasets()

    print("Dataset:", info.name)
    print("Number of classes:", NUM_CLASSES)

    datasets = [
        ("Training", train_dataset),
        ("Validation", validation_dataset),
        ("Test", test_dataset),
    ]

    for name, dataset in datasets:
        images, labels = next(iter(dataset))

        assert images.shape[1:] == (28, 28, 1), (
            f"{name}: unexpected image shape {images.shape}"
        )
        assert images.dtype.name == "float32", (
            f"{name}: unexpected image dtype {images.dtype}"
        )

        pixel_min = float(images.numpy().min())
        pixel_max = float(images.numpy().max())

        assert pixel_min >= 0.0, f"{name}: pixel values below 0.0"
        assert pixel_max <= 1.0, f"{name}: pixel values above 1.0"

        label_min = int(labels.numpy().min())
        label_max = int(labels.numpy().max())

        assert label_min >= 0, f"{name}: labels below 0"
        assert label_max < NUM_CLASSES, (
            f"{name}: labels must be below {NUM_CLASSES}"
        )

        print(f"\n{name}: OK")
        print("Image batch shape:", images.shape)
        print("Image dtype:", images.dtype)
        print(f"Pixel range: {pixel_min:.4f} to {pixel_max:.4f}")
        print("Label batch shape:", labels.shape)
        print(f"Label range: {label_min} to {label_max}")

    test_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])
    test_classes = np.unique(test_labels)

    assert len(test_classes) == NUM_CLASSES, (
        f"Test: expected {NUM_CLASSES} classes, found {len(test_classes)}"
    )

    counts = np.bincount(test_labels, minlength=NUM_CLASSES)
    print(f"\nTest: all {NUM_CLASSES} classes present")
    print(f"Samples per class: {counts.min()} to {counts.max()}")

    print("\nAll dataset checks passed!")


if __name__ == "__main__":
    main()
