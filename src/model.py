import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_unet(input_shape, output_channels=1, embed_dim=64, num_stages=3, kernel_size=4, strides=2):
    """Standard U-Net architecture used for WATER eddy detection and SST reconstruction."""
    inputs = keras.Input(shape=input_shape)
    x_downsample = []
    x = inputs
    x_downsample.append(x)

    # Encoder
    for stage in range(num_stages):
        initializer = tf.random_normal_initializer(0., 0.02)
        x = layers.Conv2D(embed_dim*2**stage, kernel_size, strides, activation='relu', 
                          padding='same', kernel_initializer=initializer, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x_downsample.append(x)

    # Decoder
    x_downsample = list(reversed(x_downsample[:-1]))
    for stage in range(num_stages-1):
        initializer = tf.random_normal_initializer(0., 0.02)
        x = layers.Conv2DTranspose(embed_dim*2**(num_stages-stage-2), kernel_size, strides, 
                                   activation='relu', padding='same', 
                                   kernel_initializer=initializer, use_bias=False)(x)
        x = layers.Resizing(height=x_downsample[stage].shape[1], width=x_downsample[stage].shape[2])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Concatenate()([x, x_downsample[stage]])

    # Output
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = layers.Conv2DTranspose(output_channels, kernel_size, strides, padding='same', 
                                     kernel_initializer=initializer, activation=None)(x)
    outputs = layers.Resizing(height=x_downsample[-1].shape[1], width=x_downsample[-1].shape[2])(outputs)
    
    return keras.Model(inputs=inputs, outputs=outputs)