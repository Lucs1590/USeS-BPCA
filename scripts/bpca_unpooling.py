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

    def build(self, input_shape):
        super(BPCAUnpooling, self).build(input_shape)

    @tf.function
    def bpca_unpooling(self, transformed_patches):
        # Compute the region of interest
        h, w, c = self.expected_shape  # block_height, block_width, block_channels
        d = c // (self.pool_size * self.pool_size)  # block_depth

        # Perform the reverse PCA transformation on the transformed patches
        pca_components = tf.linalg.matrix_transpose(
            tf.linalg.svd(
                transformed_patches,
                compute_uv=False
            ).V[:, :self.n_components]
        )
        original_patches = tf.matmul(transformed_patches, pca_components)

        # Revert the PCA transformation by multiplying by the standard deviation and adding the mean
        mean = tf.reduce_mean(original_patches, axis=0)
        std = tf.math.reduce_std(original_patches, axis=0)
        original_patches = (original_patches * std) + mean

        # Reconstruct the original data
        original_patches = tf.where(
            tf.math.is_nan(original_patches),
            0.0,
            original_patches
        )
        original_patches = tf.reshape(
            original_patches,
            [h // self.pool_size, w // self.pool_size, c]
        )
        return original_patches

    def call(self, inputs):
        unpooled = tf.vectorized_map(self.bpca_unpooling, inputs)
        return unpooled
