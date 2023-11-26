import tensorflow as tf


class BPCAUnpooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, stride=2, n_components=1, expected_shape=None, **kwargs):
        super(BPCAUnpooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.n_components = n_components
        self.expected_shape = expected_shape

        self.patch_size = [1, self.pool_size, self.pool_size, 1]
        self.strides = [1, self.stride, self.stride, 1]

        self.pca_components = tf.keras.layers.Dense(
            units=self.n_components * self.expected_shape[2],
            activation='tanh',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )

    def build(self, input_shape):
        super(BPCAUnpooling, self).build(input_shape)

    @tf.function
    def bpca_unpooling(self, transformed_patches):
        # Compute the region of interest
        h, w, c = self.expected_shape  # block_height, block_width, block_channels

        # Pad the input patches with zeros
        padded_patches = tf.pad(
            transformed_patches,
            [
                [0, 0],
                [self.pool_size - 1, self.pool_size - 1],
                [self.pool_size - 1, self.pool_size - 1],
                [0, 0]
            ], constant_values=0.0
        )

        # Learn the PCA components
        pca_components = self.pca_components(padded_patches)
        pca_components = tf.reshape(
            pca_components,
            [-1, h, w, self.n_components, 1]
        )

        # Reshape the transformed patches
        transformed_patches = tf.reshape(transformed_patches, [-1, h, w, c])

        # Perform the PCA decomposition
        u, s, vh = tf.linalg.svd(transformed_patches, compute_uv=True)

        # Reshape s (None, 32, 32, 32) to (None, 32, 32, 32, 1)
        s = tf.reshape(s, [-1, h, w, self.n_components, 1])

        # Multiply s (None, 32, 32, 32, 1) by pca_components (None, 32, 32, 1, 512)
        mult = tf.matmul(s, pca_components)
        mult = tf.reshape(mult, [-1, h, w, self.n_components * c])

        transformed_patches = tf.matmul(
            u[..., tf.newaxis, tf.newaxis],
            mult
        )

        # Revert the PCA transformation by multiplying by the standard deviation and adding the mean
        mean = tf.reduce_mean(transformed_patches, axis=0)
        std = tf.math.reduce_std(transformed_patches, axis=0)
        transformed_patches = (transformed_patches * std) + mean

        # Reconstruct the original data
        transformed_patches = tf.where(
            tf.math.is_nan(transformed_patches),
            0.0,
            transformed_patches
        )

        # Upsample the original patches
        transformed_patches = tf.reshape(
            transformed_patches,
            [-1, h, w, c]
        )

        # Return the output patches
        return transformed_patches

    def call(self, inputs):
        return self.bpca_unpooling(inputs)
