# import sys
# sys.path.append("/kaggle/input/library/keras/default/1")
import os
import pickle
import tqdm
import csv
import numpy as np

import Utils
import Models
import Clients

def fed_avg(grads_list, **kwargs):
    num_clients = len(grads_list)
    return [sum(layer_grads_list) / num_clients for layer_grads_list in zip(*grads_list)]

def l2_norm(grads_list, **kwargs):
    norm_threshold = 1.0
    if "norm_threshold" in kwargs:
        norm_threshold = kwargs["norm_threshold"]
    num_clients = len(grads_list)
    norm_grads_list = []
    for grads in grads_list:
        norm = 0
        for layer_grads in grads:
            norm += np.sum(layer_grads ** 2)
        norm **= 0.5
        if norm <= norm_threshold:
            norm_grads_list.append(grads)
        else:
            norm_grads_list.append([layer_grads / norm * norm_threshold for layer_grads in grads])
    return [sum(layer_grads_list) / num_clients for layer_grads_list in zip(*norm_grads_list)]

def trimmed_mean(grads_list, **kwargs):
    trimmed_ratio = 0.25
    if "trimmed_ratio" in kwargs:
        trimmed_ratio = kwargs["trimmed_ratio"]
    num_clients = len(grads_list)
    updated_grads = []
    start_idx = int(num_clients * trimmed_ratio)
    end_idx = int(-(-num_clients * (1 - trimmed_ratio) // 1))
    for layer_grads_list in zip(*grads_list):
        layer_grads_array = np.array(layer_grads_list)
        layer_grads_array.sort(axis=0)
        updated_grads.append(sum(layer_grads_array[start_idx : end_idx]) / (end_idx - start_idx))
    return updated_grads

def multi_krum(grads_list, **kwargs):
    byzantine_ratio = 0.5
    if "byzantine_ratio" in kwargs:
        byzantine_ratio = kwargs["byzantine_ratio"]
    num_clients = len(grads_list)
    cand_grads_list = []
    cand_cnt = num_clients - int(num_clients * byzantine_ratio)
    score_list = []
    for i in range(num_clients):
        dist_list = []
        for j in range(num_clients):
            norm = 0
            for layer_grads_i, layer_grads_j in zip(grads_list[i], grads_list[j]):
                norm += np.sum((layer_grads_i - layer_grads_j) ** 2)
            dist_list.append(norm)
        dist_list.sort()
        score_list.append(sum(dist_list[:cand_cnt]))
    sorted_idx_list = sorted(range(num_clients), key=lambda k: score_list[k])
    for idx in sorted_idx_list[:cand_cnt]:
        cand_grads_list.append(grads_list[idx])
    return [sum(layer_grads_list) / num_clients for layer_grads_list in zip(*cand_grads_list)]

class FederatedLearning():
    def __init__(self, input_dim, output_dim, clients: list[Clients.Client], test_dataset, has_backdoor=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clients = clients
        self.test_dataset = test_dataset
        self.has_backdoor = has_backdoor
        if has_backdoor:
            self.backdoor_test_dataset = []
            mask = np.ones(len(test_dataset[0][0]))
            mask[:len(test_dataset[0][0]) // 16] = 0
            self.backdoor_test_dataset.append([feature * mask + 0.5 * (1 - mask) for feature in test_dataset[0]])
            self.backdoor_test_dataset.append([label for label in test_dataset[1]])
        self.num_ordinary = 0
        self.num_anomaly = 0
        for client in self.clients:
            if client.is_anomaly():
                self.num_anomaly += 1
            else:
                self.num_ordinary += 1
        self.num_clients = self.num_ordinary + self.num_anomaly

    def get_pred(self, features):
        return self.clients[0].pred(features)[1]
    
    def save_states(self, path):
        with open(path + ".pickle", 'wb') as file:
            pickle.dump(self.clients, file)
    
    def save_results(self, path, r, num_round):
        if not os.path.exists(path + ".csv"):
            with open(path + ".csv", 'w', newline='') as file:
                writer = csv.writer(file)
                if self.has_backdoor:
                    writer.writerow(["round",
                                     "test_loss", "test_acc",
                                     "test_macro_f1", "test_macro_recall", "test_macro_precision",
                                     "test_micro_f1", "test_micro_recall", "test_micro_precision",
                                     "test_weighted_f1", "test_weighted_recall", "test_weighted_precision",
                                     "test_loss_backdoor", "test_acc_backdoor",
                                     "test_macro_f1_backdoor", "test_macro_recall_backdoor", "test_macro_precision_backdoor",
                                     "test_micro_f1_backdoor", "test_micro_recall_backdoor", "test_micro_precision_backdoor",
                                     "test_weighted_f1_backdoor", "test_weighted_recall_backdoor", "test_weighted_precision_backdoor"]) 
                else:
                    writer.writerow(["round",
                                     "test_loss", "test_acc",
                                     "test_macro_f1", "test_macro_recall", "test_macro_precision",
                                     "test_micro_f1", "test_micro_recall", "test_micro_precision",
                                     "test_weighted_f1", "test_weighted_recall", "test_weighted_precision"]) 
        metrics = Utils.get_metrics(np.array(self.test_dataset[1]), self.get_pred(self.test_dataset[0]))
        if self.has_backdoor:
            backdoor_metrics = Utils.get_metrics(np.array(self.backdoor_test_dataset[1]), self.get_pred(self.backdoor_test_dataset[0]))
        else:
            backdoor_metrics = []
        with open(path + ".csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([r] + metrics + backdoor_metrics)
        if self.has_backdoor:
            print(f"Save to {path + '.csv'}: {r} / {num_round} - test_loss: {metrics[0]}, test_acc: {metrics[1]}, "
                  f"test_loss_backdoor: {backdoor_metrics[0]}, test_acc_backdoor: {backdoor_metrics[1]}")
        else:
            print(f"Save to {path + '.csv'}: {r} / {num_round} - test_loss: {metrics[0]}, test_acc: {metrics[1]}")

    def run(self, num_round, num_epoch, aggr_func=fed_avg, start_round=1, **kwargs):
        # initialization phase
        current_weights = self.clients[0].model.get_weights()
        for client in self.clients:
            client.model.set_weights(current_weights)
        
        # distributed learning phase
        for r in range(start_round, num_round + 1):
            # training phase
            for client in tqdm.tqdm(self.clients):
                client.train(epochs=num_epoch)

            # aggregation phase
            grads_list = [[layer_weights - current_layer_weights for current_layer_weights, layer_weights in zip(current_weights, client.model.get_weights())] for client in self.clients]
            updated_grads = aggr_func(grads_list, **kwargs)
            current_weights = [current_layer_weights + layer_grads for current_layer_weights, layer_grads in zip(current_weights, updated_grads)]

            # synchronization phase
            for client in self.clients:
                client.model.set_weights(current_weights)

            # evaluation phase
            self.save_results(f"FLSave_{aggr_func.__name__}_Results", r, num_round)
            self.save_states(f"FLSave_{aggr_func.__name__}")


if __name__ == "__main__":
    is_iid = True
    num_round = 20
    num_epoch = 10
    num_ordinary_clients = 90
    num_anomaly_clients = 10
    num_clients = num_ordinary_clients + num_anomaly_clients
    lr = 1e-3
    with open("/kaggle/input/cifar100-pretrained-coatnet-cifar10-feature/CoAtNetFeatures_Train.pickle", 'rb') as file:
        train_features = pickle.load(file)
    with open("/kaggle/input/cifar100-pretrained-coatnet-cifar10-feature/CoAtNetFeatures_Test.pickle", 'rb') as file:
        test_features = pickle.load(file)
    input_dim = len(train_features[0][0])
    output_dim = len(train_features)
    num_dataset_per_client = len(train_features[0]) * output_dim // num_clients
    if is_iid:
        train_dataset = Utils.features2random_dataset(train_features)
        test_dataset = Utils.features2random_dataset(test_features)
    else:
        train_dataset = Utils.features2cycled_dataset(train_features, cycle_cnt=num_clients // output_dim)
        test_dataset = Utils.features2cycled_dataset(test_features, cycle_cnt=num_clients // output_dim)
    clients = [Clients.OrdinaryClient(Models.nn_model(input_dim, input_dim * 2, output_dim, lr), [d[i * num_dataset_per_client : (i + 1) * num_dataset_per_client] for d in train_dataset]) for i in range(num_ordinary_clients)]
    clients += [Clients.DirtyLabelClient(Models.nn_model(input_dim, input_dim * 2, output_dim, lr), [d[i * num_dataset_per_client : (i + 1) * num_dataset_per_client] for d in train_dataset]) for i in range(num_ordinary_clients, num_clients)]
    # with open("/kaggle/input/flpacket/FLSave_fed_avg.pickle", 'rb') as file:
    #     clients = pickle.load(file)
    fl = FederatedLearning(input_dim, output_dim, clients, test_dataset)
    fl.run(num_round, num_epoch)