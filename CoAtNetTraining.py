# import sys
# sys.path.append("/kaggle/input/coatnet_weights/keras/default/1")
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "jax"
import datetime
import keras
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import f1_score, recall_score, precision_score

import Models

# Hyperparameters
IMAGE_SIZE = (224, 224, 3)
CLASSES = 100
INIT_EPOCH = 0
EPOCHS = 200
WARMUP_EPOCHS = 10
BATCH_SIZE = 64
STRAT_LR = 1e-3
END_LR = 5e-5
WD_RATE = 1e-8
CLIP_NORM = 1.0
LBL_SMOOTH = 0.1
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
assert INIT_EPOCH < EPOCHS
assert EPOCHS - WARMUP_EPOCHS >= 10
assert STRAT_LR >= END_LR

# Transfer Learning Settings
TRANSFER_LEARNING = False
FINETUNING = False
SAME_RESOLUTION = True
WEIGHTS_PATH = None
TRANSFER_WEIGHTS_PATH = None
assert not TRANSFER_LEARNING or WEIGHTS_PATH is not None or TRANSFER_WEIGHTS_PATH is not None
assert TRANSFER_LEARNING or not FINETUNING
assert SAME_RESOLUTION or FINETUNING # When the resolution of the fine-tuning and pre-training datasets are different, the pre-trained model must be unfrozen(fine-tuned)

# System Settings
AUTOTUNE = tf.data.AUTOTUNE
TIME = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(Metrics, self).__init__()
        self.val_data = list(val_data)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_pred, val_true = keras.ops.convert_to_tensor([]), keras.ops.convert_to_tensor([])
        for X, y in self.val_data:
            val_pred = keras.ops.append(val_pred, keras.ops.argmax(self.model.predict(X, verbose=0), -1))
            val_true = keras.ops.append(val_true, keras.ops.argmax(y, -1))

        val_macro_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        val_macro_recall = recall_score(val_true, val_pred, average="macro", zero_division=0)
        val_macro_precision = precision_score(val_true, val_pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = val_macro_f1
        logs["val_macro_recall"] = val_macro_recall
        logs["val_macro_precision"] = val_macro_precision
        
        val_micro_f1 = f1_score(val_true, val_pred, average="micro", zero_division=0)
        val_micro_recall = recall_score(val_true, val_pred, average="micro", zero_division=0)
        val_micro_precision = precision_score(val_true, val_pred, average="micro", zero_division=0)
        logs["val_micro_f1"] = val_micro_f1
        logs["val_micro_recall"] = val_micro_recall
        logs["val_micro_precision"] = val_micro_precision

        val_weighted_f1 = f1_score(val_true, val_pred, average="weighted", zero_division=0)
        val_weighted_recall = recall_score(val_true, val_pred, average="weighted", zero_division=0)
        val_weighted_precision = precision_score(val_true, val_pred, average="weighted", zero_division=0)
        logs["val_weighted_f1"] = val_weighted_f1
        logs["val_weighted_recall"] = val_weighted_recall
        logs["val_weighted_precision"] = val_weighted_precision

def lr_scheduler(epoch, lr):
    if epoch < WARMUP_EPOCHS:
        lr_rate = STRAT_LR * (epoch + 1) / WARMUP_EPOCHS
    else:
        decay_step = (epoch - WARMUP_EPOCHS) // 5
        decay_steps = (EPOCHS - WARMUP_EPOCHS) // 5
        lr_rate = STRAT_LR / (STRAT_LR / END_LR) ** (decay_step / (decay_steps - 1))
    return lr_rate

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE[0:2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, CLASSES)
    return {"images": image, "labels": label}

def prepare_dataset(dataset, split):
    if split == "train":
        return dataset.shuffle(10 * BATCH_SIZE).map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

def apply_augment(inputs):
    rand_aug = keras_cv.layers.RandAugment((0, 1), 2, 0.5)
    mix_up = keras_cv.layers.MixUp(0.2)
    inputs["images"] = rand_aug(inputs["images"])
    inputs = mix_up(inputs, training=True)
    return inputs

def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = (images - MEAN) / STD
    return images, labels

keras.backend.clear_session()
tfds.disable_progress_bar()

mode_str = ""
if TRANSFER_LEARNING:
    # Dataset
    train, test = tfds.load("cifar10", split=["train","test"], as_supervised=True)
    train_dataset = prepare_dataset(train, "train").map(apply_augment, num_parallel_calls=AUTOTUNE).map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_dataset = prepare_dataset(test, "test").map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # Pre-trained Model
    pretrain_model = Models.CoAtNet.coatnet0(IMAGE_SIZE)
    if WEIGHTS_PATH is not None and TRANSFER_WEIGHTS_PATH is None:
        if SAME_RESOLUTION:
            pretrain_model.load_weights(WEIGHTS_PATH)
        else:
            pretrain_model.load_weights(WEIGHTS_PATH, skip_mismatch=True)
    if FINETUNING:
        mode_str += "FineTuning_"
    else:
        mode_str += "TransferLearning_"
        pretrain_model.trainable = False
    
    # Model
    inputs = keras.layers.Input(IMAGE_SIZE)
    hidden = pretrain_model(inputs, training=False)
    hidden = keras.layers.GlobalAveragePooling2D()(hidden)
    outputs = keras.layers.Dense(CLASSES, activation=keras.activations.softmax)(hidden)
    model = keras.Model(inputs, outputs)
    if TRANSFER_WEIGHTS_PATH is not None:
        model.load_weights(TRANSFER_WEIGHTS_PATH)
else:
    mode_str += "PreTraining_"
    # Dataset
    train, test = tfds.load("cifar100", split=["train","test"], as_supervised=True)
    train_dataset = prepare_dataset(train, "train").map(apply_augment, num_parallel_calls=AUTOTUNE).map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_dataset = prepare_dataset(test, "test").map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    # Model
    model = Models.CoAtNet.coatnet0(IMAGE_SIZE, CLASSES)
    if WEIGHTS_PATH is not None:
        model.load_weights(WEIGHTS_PATH)

mode_str += f"{IMAGE_SIZE[0]}_{CLASSES}_{TIME}"
model_callback = keras.callbacks.ModelCheckpoint(f"./CoAtNet_Model_{mode_str}.weights.h5", verbose=1, save_best_only=True, save_weights_only=True, mode="min")
history_callback = keras.callbacks.CSVLogger(f"./CoAtNet_History_{mode_str}.csv", separator=',', append=True)
lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
metrics_callback = Metrics(test_dataset)

model.summary(show_trainable=True)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=STRAT_LR, weight_decay=WD_RATE, clipnorm=CLIP_NORM), 
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LBL_SMOOTH), 
    metrics=[keras.metrics.categorical_accuracy]
)
model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, verbose=2, callbacks=[metrics_callback, lr_callback, model_callback, history_callback], initial_epoch=INIT_EPOCH)