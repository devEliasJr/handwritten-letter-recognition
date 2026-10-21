
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME = "emnist/letters"
NUM_CLASSES = 26
IMAGE_SIZE = 28


def preprocess_image(image: tf.Tensor, label: tf.Tensor):
    """Prepare an EMNIST image and its label for CNN training."""

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int64) - 1

    return image, label


def load_datasets(batch_size: int = 128):
    """Load and prepare training, validation, and test datasets."""
    splits, info = tfds.load(
        DATASET_NAME,
        split=[
            "train[:80%]",
            "train[80%:90%]",
            "train[90%:]",
        ],
        as_supervised=True,
        with_info=True,
    )

    train_raw, validation_raw, test_raw = splits

    def prepare(dataset, training: bool = False):
        dataset = dataset.map(
            preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.cache()

        if training:
            dataset = dataset.shuffle(
                buffer_size=10_000,
                reshuffle_each_iteration=True,
            )

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_dataset = prepare(train_raw, training=True)
    validation_dataset = prepare(validation_raw)
    test_dataset = prepare(test_raw)

    return train_dataset, validation_dataset, test_dataset, info
