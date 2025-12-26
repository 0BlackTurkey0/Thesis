import random
import keras
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

# IID
def features2random_dataset(features):
    data = []
    labels = []
    for label in range(len(features)):
        data += list(features[label])
        labels += [list(keras.utils.to_categorical(label, num_classes=len(features))) for _ in range(len(features[label]))]
    dataset = list(zip(data, labels))
    random.shuffle(dataset)
    return list(map(list, zip(*dataset)))

# Non-IID
def features2cycled_dataset(features, cycle_cnt):
    data = []
    labels = []
    data_cnt_per_cycle_class = len(features[0]) // cycle_cnt
    for cycle in range(cycle_cnt):
        start_idx = data_cnt_per_cycle_class * cycle
        for label in range(len(features)):
            data += list(features[label][start_idx : start_idx + data_cnt_per_cycle_class])
            labels += [keras.utils.to_categorical(label, num_classes=len(features)) for _ in range(data_cnt_per_cycle_class)]
    dataset = list(zip(data, labels))
    return list(map(list, zip(*dataset)))

# Metrics
def get_metrics(true_one_hot, pred_one_hot):
    cce = keras.losses.CategoricalCrossentropy()
    loss = np.squeeze(cce(true_one_hot, pred_one_hot))
    true = np.argmax(true_one_hot, axis=-1)
    pred = np.argmax(pred_one_hot, axis=-1)
    acc = keras.metrics.Accuracy()
    acc.update_state(true, pred)
    accuracy = np.squeeze(acc.result())
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    macro_recall = recall_score(true, pred, average="macro", zero_division=0)
    macro_precision = precision_score(true, pred, average="macro", zero_division=0)
    micro_f1 = f1_score(true, pred, average="micro", zero_division=0)
    micro_recall = recall_score(true, pred, average="micro", zero_division=0)
    micro_precision = precision_score(true, pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(true, pred, average="weighted", zero_division=0)
    weighted_precision = precision_score(true, pred, average="weighted", zero_division=0)
    return [loss, accuracy, macro_f1, macro_recall, macro_precision, micro_f1, micro_recall, micro_precision, weighted_f1, weighted_recall, weighted_precision]