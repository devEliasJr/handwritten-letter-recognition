
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def label_to_letter(label):
    return chr(ord("A") + int(label) - 1)


def main():
    dataset, info = tfds.load(
        "emnist/letters",
        split="train",
        as_supervised=True,
        with_info=True,
    )

    print("Training examples:", info.splits["train"].num_examples)
    print("Observed labels: 1-26 (A-Z)")

    samples = list(dataset.take(5))

    figure, axes = plt.subplots(2, 5, figsize=(12, 5))

    for index, (image, label) in enumerate(samples):
        image = image.numpy().squeeze()
        letter = label_to_letter(label.numpy())

        axes[0, index].imshow(image, cmap="gray")
        axes[0, index].set_title(f"Original: {letter}")
        axes[0, index].axis("off")

        axes[1, index].imshow(image.T, cmap="gray")
        axes[1, index].set_title(f"Transposed: {letter}")
        axes[1, index].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
