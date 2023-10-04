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
    # first part of the U - contracting part
    c0 = layers.Conv2D(64, activation='relu', kernel_size=3)(inputs)
    c1 = layers.Conv2D(64, activation='relu', kernel_size=3)(
        c0)  # This layer for concatenating in the expansive part
    c2 = layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='valid')(c1)

    c3 = layers.Conv2D(128, activation='relu', kernel_size=3)(c2)
    c4 = layers.Conv2D(128, activation='relu', kernel_size=3)(
        c3)  # This layer for concatenating in the expansive part
    c5 = layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='valid')(c4)

    c6 = layers.Conv2D(256, activation='relu', kernel_size=3)(c5)
    c7 = layers.Conv2D(256, activation='relu', kernel_size=3)(
        c6)  # This layer for concatenating in the expansive part
    c8 = layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='valid')(c7)

    c9 = layers.Conv2D(512, activation='relu', kernel_size=3)(c8)
    c10 = layers.Conv2D(512, activation='relu', kernel_size=3)(
        c9)  # This layer for concatenating in the expansive part
    c11 = layers.MaxPool2D(pool_size=(
        2, 2), strides=(2, 2), padding='valid')(c10)

    c12 = layers.Conv2D(1024, activation='relu', kernel_size=3)(c11)
    c13 = layers.Conv2D(1024, activation='relu',
                        kernel_size=3, padding='valid')(c12)

    # We will now start the second part of the U - expansive part
    t01 = layers.Conv2DTranspose(
        512, kernel_size=2, strides=(2, 2), activation='relu')(c13)
    crop01 = layers.Cropping2D(cropping=(4, 4))(c10)

    concat01 = layers.concatenate([t01, crop01], axis=-1)

    c14 = layers.Conv2D(512, activation='relu', kernel_size=3)(concat01)
    c15 = layers.Conv2D(512, activation='relu', kernel_size=3)(c14)

    t02 = layers.Conv2DTranspose(
        256, kernel_size=2, strides=(2, 2), activation='relu')(c15)
    crop02 = layers.Cropping2D(cropping=(16, 16))(c7)

    concat02 = layers.concatenate([t02, crop02], axis=-1)

    c16 = layers.Conv2D(256, activation='relu', kernel_size=3)(concat02)
    c17 = layers.Conv2D(256, activation='relu', kernel_size=3)(c16)

    t03 = layers.Conv2DTranspose(
        128, kernel_size=2, strides=(2, 2), activation='relu')(c17)
    crop03 = layers.Cropping2D(cropping=(40, 40))(c4)

    concat03 = layers.concatenate([t03, crop03], axis=-1)

    c18 = layers.Conv2D(128, activation='relu', kernel_size=3)(concat03)
    c19 = layers.Conv2D(128, activation='relu', kernel_size=3)(c18)

    t04 = layers.Conv2DTranspose(
        64, kernel_size=2, strides=(2, 2), activation='relu')(c19)
    crop04 = layers.Cropping2D(cropping=(88, 88))(c1)

    concat04 = layers.concatenate([t04, crop04], axis=-1)

    c20 = layers.Conv2D(64, activation='relu', kernel_size=3)(concat04)
    c21 = layers.Conv2D(64, activation='relu', kernel_size=3)(c20)

    outputs = layers.Conv2D(num_classes, kernel_size=3,
                            activation='softmax', padding="same")(c21)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def unet_model(img_size, num_classes):
    def convolution_block(input, num_filters):
        x = layers.Conv2D(num_filters, 3, padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(num_filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return x

    def encoder_block(input, num_filters):
        conv_layer = convolution_block(input, num_filters)
        pooling = layers.MaxPooling2D((2, 2))(conv_layer)
        return conv_layer, pooling

    def decoder_block(input, skip_features, num_filters):
        x = layers.Conv2DTranspose(
            num_filters, (2, 2), strides=2, padding='same')(input)
        x = layers.Concatenate()([x, skip_features])
        x = convolution_block(x, num_filters)
        return x

    inputs = layers.Input(shape=img_size + (3,))
    filters = [64, 128, 256, 512]
    saved_layers = []

    pooling = inputs
    for filter in filters:
        conv_layer, pooling = encoder_block(pooling, filter)
        saved_layers.append(conv_layer)

    conv_block = convolution_block(pooling, 1024)

    deconv_layer = conv_block
    for filter in reversed(filters):
        conv_layer = saved_layers.pop()
        deconv_layer = decoder_block(deconv_layer, conv_layer, filter)

    outputs = layers.Conv2D(num_classes, 3, padding='same',
                            activation='softmax')(deconv_layer)

    model = keras.Model(inputs, outputs, name='UNet')
    return model


HEIGHT, WIDTH = 256, 256
NUM_CLASSES = 3  # background, foreground, boundary
model = unet_model(img_size=(HEIGHT, WIDTH), num_classes=NUM_CLASSES)
model.summary()
