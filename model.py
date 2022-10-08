import tensorflow as tf

def get_model():
    base_model = tf.keras.applications.InceptionV3(include_top=False, input_shape=(128, 128, 3), weights="imagenet")
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(128)(x)

    model = tf.keras.Model(inputs, outputs)
    return model
