# import sys
# sys.path.append("/kaggle/input/coatnet_pretrain_cifar100_weights/keras/default/1")
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import pickle
import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

import Models

WEIGHTS_PATH = "/kaggle/input/coatnet_pretrain_cifar100/keras/default/1/Models/CoAtNet_Model_PreTraining_224_100_2024_11_04_04_35_22.weights.h5"
IMAGE_SIZE = (224, 224, 3)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
AUTOTUNE = tf.data.AUTOTUNE

features_train_list = [[] for _ in range(10)]
features_test_list = [[] for _ in range(10)]

def process_for_model(image, label):
    image = tf.image.resize(image, IMAGE_SIZE[0:2])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train, test = tfds.load("cifar10", split=["train","test"], as_supervised=True)
train_ds = train.map(process_for_model, num_parallel_calls=AUTOTUNE).batch(50)
test_ds = test.map(process_for_model, num_parallel_calls=AUTOTUNE).batch(50)
pretrain_model = Models.CoAtNet.coatnet0(IMAGE_SIZE)
pretrain_model.load_weights(WEIGHTS_PATH)
inputs = keras.layers.Input(IMAGE_SIZE)
hidden = pretrain_model(inputs, training=False)
outputs = keras.layers.GlobalAveragePooling2D()(hidden)
model = keras.Model(inputs, outputs)

for images, labels in tqdm.tqdm(train_ds):
    features = model.predict(images, verbose=0)
    lbls = labels.numpy()
    for i in range(50):
        features_train_list[lbls[i].item()].append(features[i])
features_train_array = np.array(features_train_list)
with open("CoAtNetFeatures_Train.pickle", 'wb') as file:
    pickle.dump(features_train_array, file)

for images, labels in tqdm.tqdm(test_ds):
    features = model.predict(images, verbose=0)
    lbls = labels.numpy()
    for i in range(50):
        features_test_list[lbls[i].item()].append(features[i])
features_test_array = np.array(features_test_list)
with open("CoAtNetFeatures_Test.pickle", 'wb') as file:
    pickle.dump(features_test_array, file)

# with open("/kaggle/working/CoAtNetFeatures_Train.pickle", 'rb') as file:
#     data = pickle.load(file)
# print(data.shape)
# with open("/kaggle/working/CoAtNetFeatures_Test.pickle", 'rb') as file:
#     data = pickle.load(file)
# print(data.shape)