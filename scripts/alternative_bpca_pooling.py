import tensorflow as tf


class BPCAPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, stride=2, n_components=1, expected_shape=None, **kwargs):
        super(BPCAPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.n_components = n_components
        self.expected_shape = expected_shape

        self.patch_size = [1, self.pool_size, self.pool_size, 1]
        self.strides = [1, self.stride, self.stride, 1]

    def build(self, input_shape):
        super(BPCAPooling, self).build(input_shape)

    @tf.function
    def bpca_pooling(self, feature_map):
        # Compute the region of interest
        h, w, c = self.expected_shape  # block_height, block_width, block_channels
        d = c // (self.pool_size * self.pool_size)  # block_depth

        # Create blocks (patches)
        data = tf.reshape(feature_map, [1, h, w, c])
        patches = tf.image.extract_patches(
            images=data,
            sizes=self.patch_size,
            strides=self.strides,
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            [h*w*d, self.pool_size * self.pool_size]
        )

        # PCA
        # Centering the data
        mean = tf.reduce_mean(patches, axis=0)
        centered_patches = patches - mean

        # Compute covariance matrix
        cov_matrix = tf.matmul(centered_patches, centered_patches, transpose_a=True) / \
            tf.cast(tf.shape(centered_patches)[0], tf.float32)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = tf.linalg.eigh(cov_matrix)

        # Get top 'n_components' eigenvectors
        top_eigenvectors = eigenvectors[:, -self.n_components:]

        # Project patches onto the principal components
        transformed_patches = tf.matmul(centered_patches, top_eigenvectors)

        return tf.reshape(transformed_patches, [h//self.stride, w//self.stride, d * self.stride * self.stride])

    def call(self, inputs):
        pooled = tf.map_fn(self.bpca_pooling, inputs)
        return pooled
