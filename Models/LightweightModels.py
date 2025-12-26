import keras

def nn_model(input_dim, latent_dim, output_dim, lr):
    inputs = keras.layers.Input((input_dim,))
    hidden = keras.layers.Dense(latent_dim)(inputs)
    outputs = keras.layers.Dense(output_dim, activation=keras.activations.softmax)(hidden)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr), 
        loss=keras.losses.CategoricalCrossentropy(), 
        metrics=[keras.metrics.categorical_accuracy]
    )
    return model

def nn_binary_model(input_dim, latent_dim, latent2_dim, lr):
    inputs = keras.layers.Input((input_dim,))
    hidden = keras.layers.Dense(latent_dim)(inputs)
    hidden2 = keras.layers.Dense(latent2_dim)(hidden)
    outputs = keras.layers.Dense(1, activation=keras.activations.sigmoid)(hidden2)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr), 
        loss=keras.losses.BinaryCrossentropy(), 
        metrics=[keras.metrics.binary_accuracy]
    )
    return model