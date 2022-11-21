from . import util
import tensorflow as tf

# Create function to train model
def train(model, X_train, y_train, batch_size=128, epochs=20):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return model

# Create function to instantiate, build model and train data
def fit(data, loss="mse", learning_rate=0.01):
    columns = data.columns

    X_train = data.drop(util.TARGET, axis=1).values
    y_train = data[util.TARGET].values

    print(X_train.shape, y_train.shape)

    model = tf.keras.Sequential([   
        tf.keras.layers.Input(shape=(X_train.shape[1], )),
        tf.keras.layers.Dense(13, activation="relu"),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(X_train.shape[1])
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)
    print(model.summary())

    model = train(model, X_train, X_train)

    return model, columns