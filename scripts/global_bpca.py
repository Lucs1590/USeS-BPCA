import tensorflow as tf

class GlobalBPCAPooling2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalBPCAPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GlobalBPCAPooling2D, self).build(input_shape)

    @tf.function
    def bpca_pooling(self, feature_map):
        # Compute the region of interest
        h, w, c = feature_map.shape  # block_height, block_width, block_channels
        pool_size = h
        patch_size = [1, pool_size, pool_size, 1]
        strides = [1, pool_size, pool_size, 1]
        
        # Create blocks (patches)
        data = tf.reshape(feature_map, [1, h, w, c])
        patches = tf.image.extract_patches(
            images=data,
            sizes=patch_size,
            strides=strides,
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, [-1, pool_size * pool_size])

        # Normalize the data by subtracting the mean and dividing by the standard deviation
        mean = tf.reduce_mean(patches, axis=0)
        std = tf.math.reduce_std(patches, axis=0)
        patches = (patches - mean) / std
        patches = tf.where(tf.math.is_nan(patches), 0.0, patches)

        # Perform the Singular Value Decomposition (SVD) on the data
        _, _, v = tf.linalg.svd(patches)

        # Extract the first principal component from the matrix v
        pca_components = v[:, :1]

        # Perform the PCA transformation on the data
        transformed_patches = tf.matmul(patches, pca_components)

        return tf.reshape(transformed_patches, [h // pool_size, w // pool_size, c])

    def call(self, inputs):
        pooled = tf.vectorized_map(self.bpca_pooling, inputs)
        pooled = tf.reshape(pooled, [-1, pooled.shape[-1]])
        return pooled
