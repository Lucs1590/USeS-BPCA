import random
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    # color augmentation
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_saturation(input_image, 0, 2)
        input_image = tf.image.random_brightness(input_image, 0.5)
        input_image = tf.image.random_contrast(input_image, 0, 2)

    # rotation augmentation
    if tf.random.uniform(()) > 0.5:
        random_degree = random.uniform(-25, 25)
        input_image = tf.image.rot90(input_image, k=int(random_degree // 90))
        input_mask = tf.image.rot90(input_mask, k=int(random_degree // 90))

    return input_image, input_mask


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def resize(input_image, input_mask):
    input_image = tf.image.resize(
        input_image,
        (256, 256),
        method="nearest"
    )
    input_mask = tf.image.resize(input_mask, (256, 256), method="nearest")

    return input_image, input_mask


def load_image_train(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = augment(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def plot_augmented_images(dataset):
    for image, mask in dataset.take(3):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()


if __name__ == "__main__":
    dataset, info = tfds.load(
        'oxford_iiit_pet:3.*.*',
        with_info=True,
        shuffle_files=True,
        split=['train[:2%]'],
        data_dir='/Volumes/SSD/datasets',
    )
    train_dataset = dataset[0].map(
        load_image_train,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    plot_augmented_images(train_dataset)
    print("Done")
