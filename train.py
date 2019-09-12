from comet_ml import Experiment
experiment = Experiment(api_key="La6Dt4UTNQ6moZ5klCeCI2bzV",	
                        project_name="general", workspace="kongkip")

import tensorflow as tf
import os
import mute_tf_warnings as mw
import matplotlib.pyplot as plt
mw.tf_mute_warning()

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def scale(image, label):
    image = tf.reshape(image, (28,28,1))
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(scale).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(scale).batch(32)



with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28,1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model.fit(train_dataset, epochs=20, validation_data=test_dataset, callbacks=[experiment.get_callback('keras')])
