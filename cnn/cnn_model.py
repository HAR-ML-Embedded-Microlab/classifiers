import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential([
    layers.Input(
        shape=(6,50)
    ),

    #layers.Normalization(),

    layers.Conv1D(
        32,
        6, 
        padding='same',
        activation='relu'
    ),

    layers.MaxPool1D(
        pool_size=3,
        strides=1,
        padding='valid'
    ),

    # layers.Dropout(0.4),

    layers.BatchNormalization(),

    layers.Conv1D(
        32,
        12, 
        padding='same',
        activation='relu'
    ),

    layers.MaxPool1D(
        pool_size=3,
        strides=1,
        padding='valid'
    ),

    # layers.Dropout(0.4),

    layers.BatchNormalization(),

    layers.Conv1D(
        32,
        12, 
        padding='same',
        activation='relu'
    ),

    layers.MaxPool1D(
        pool_size=3,
        strides=1,
        padding='valid'
    ),

    # layers.Dropout(0.4),

    layers.BatchNormalization(),

    layers.Conv1D(
        32,
        32, 
        padding='same',
        activation='relu'
    ),

    layers.MaxPool1D(
        pool_size=3,
        strides=1,
        padding='valid'
    ),

    # layers.Dropout(0.4),

    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(
        512,
        activation='relu'
    ),

    layers.Dense(
        512,
        activation='relu'
    ),

    # layers.Dropout(0.2),

    layers.Dense(
        12,
        activation='softmax'
    )
])


print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3,
                                      momentum=0.9), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

model.save('work/model')

