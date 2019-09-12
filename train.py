import tensorflow as tf
import os
import mute_tf_warnings as mw

mw.tf_mute_warning()

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(scale).batch(32)


test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = train_dataset.map(scale).batch(32)

strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
