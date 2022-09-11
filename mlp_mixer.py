import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def Serf(x):
  return x*tf.math.erf(tf.math.softplus(x))


class Patches(layers.Layer):
    def __init__(self, patch_size, num_patches, *args, **kwargs):
        super(Patches, self).__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches


class MLPMixerLayer(layers.Layer):
    def __init__(self, S, C, DS, DC, dropout_rate, *args, **kwargs):
        super(MLPMixerLayer, self).__init__(*args, **kwargs)

        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=DS),
                keras.layers.Lambda(Serf),
                layers.Dense(units=S),
                layers.Dropout(rate=dropout_rate),
            ]
        )

        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=DC),
                keras.layers.Lambda(Serf),
                layers.Dense(units=C),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize1(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = tf.linalg.matrix_transpose(x)
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = tf.linalg.matrix_transpose(mlp1_outputs)
        # Add skip connection.
        x = mlp1_outputs + inputs
        # Apply layer normalization.
        x_patches = self.normalize2(x)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(x_patches)
        # Add skip connection.
        x = x + mlp2_outputs
        return x


def MLPMixer(input_shape, num_classes, N=8, P=16, C=512, DS=256, DC=2048, dropout_rate=0):
    H, W, _ = input_shape
    S = (H*W)//(P**2)

    inputs = layers.Input(shape=input_shape, name='Input')
    # Normalize inputs.
    normalization = layers.Normalization(name='Normalization')(inputs)
    # Create patches.
    patches = Patches(P, S, name='Patches')(normalization)
    # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
    x = layers.Dense(units=C, name='Projection')(patches)
    # Process x using the module blocks.
    for i in range(1, N+1):
      x = MLPMixerLayer(S, C, DS, DC, dropout_rate, name=f'Mixer_Layer_{i}')(x)
    x = layers.LayerNormalization(epsilon=1e-6, name='Pre_head_layer_norm')(x)
    # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
    representation = layers.GlobalAveragePooling1D(name='Global_Average_Pooling')(x)
    # Apply dropout.
    representation = layers.Dropout(rate=dropout_rate, name='Drop_out')(representation)
    # Compute logits outputs.
    logits = layers.Dense(num_classes, activation='softmax', name='Class')(representation)
    # Create the Keras model.
    return keras.Model(inputs=inputs, outputs=logits)