import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from scripts.bpca_pooling import BPCAPooling


def define_crop_values(filters):
    if filters == 64:
        crop = (88, 88)
    elif filters == 128:
        crop = (40, 40)
    elif filters == 256:
        crop = (16, 16)
    elif filters == 512:
        crop = (4, 4)
    else:
        crop = (0, 0)
    return crop


def get_layers_number(number):
    if number == 64:
        return (128, 128, 64)
    if number == 128:
        return (64, 64, 128)
    if number == 256:
        return (32, 32, 256)
    if number == 512:
        return (16, 16, 512)
    else:
        return (0, 0, 0)


def get_unet_model(img_size, num_classes):
    inputs = layers.Input(shape=img_size + (3,))
    # Test the following lines (until the loop) to see if the model works
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Activation("relu")(x)

    ### [First half of the network: downsampling inputs] ###
    concatenatenating_layers = []
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, activation='relu', kernel_size=3)(x)
        x = layers.Conv2D(filters, activation='relu', kernel_size=3)(x)
        concatenatenating_layers.append(x)
        x = layers.MaxPool2D(pool_size=(
            2, 2), strides=(2, 2), padding='valid')(x)
        # x = BPCAPooling(pool_size=2, stride=2, expected_shape=get_layers_number(filters))(x)

    x = layers.Conv2D(1024, activation='relu', kernel_size=3)(x)
    x = layers.Conv2D(1024, activation='relu',
                      kernel_size=3, padding='valid')(x)

    ### [Second half of the network: upsampling inputs] ###
    for filters in [512, 256, 128]:
        transp = layers.Conv2DTranspose(
            filters, kernel_size=2, strides=(2, 2), activation='relu')(x)
        conc_conv = concatenatenating_layers.pop()
        crop = layers.Cropping2D(
            cropping=define_crop_values(filters))(conc_conv)
        conc = layers.concatenate([transp, crop], axis=-1)

        x = layers.Conv2D(filters, activation='relu', kernel_size=3)(conc)
        x = layers.Conv2D(filters, activation='relu', kernel_size=3)(x)

    outputs = layers.Conv2D(num_classes, kernel_size=3,
                            activation='softmax', padding="same")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


HEIGHT, WIDTH = 256, 256
NUM_CLASSES = 3  # background, foreground, boundary
model = get_unet_model(img_size=(HEIGHT, WIDTH), num_classes=NUM_CLASSES)
model.summary()
