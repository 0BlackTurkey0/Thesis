# import sys
# sys.path.append("/kaggle/input/library/keras/default/1")
import os
import random
import pickle
import tqdm
import csv
import numpy as np

import Utils
import Models
import Clients
import Simulators

class EnsembleDistributedLearning():
    def __init__(self, input_dim, output_dim, clients: list[Clients.Client], offline_client: Clients.Client, test_dataset, has_backdoor=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clients = clients
        self.offline_client = offline_client
        self.test_dataset = test_dataset
        self.has_backdoor = has_backdoor
        if has_backdoor:
            self.backdoor_test_dataset = []
            mask = np.ones(len(test_dataset[0][0]))
            mask[:len(test_dataset[0][0]) // 16] = 0
            self.backdoor_test_dataset.append([feature * mask + 0.5 * (1 - mask) for feature in test_dataset[0]])
            self.backdoor_test_dataset.append([label for label in test_dataset[1]])
        self.public_dataset = [[], []]
        self.num_ordinary = 0
        self.num_anomaly = 0
        self.active_clients = []
        self.dropped_clients = []
        for client in self.clients:
            if client.tokens >= batch_size:
                self.active_clients.append(client)
            else:
                self.dropped_clients.append(client)
            if client.is_anomaly():
                self.num_anomaly += 1
            else:
                self.num_ordinary += 1
        self.num_clients = self.num_ordinary + self.num_anomaly
    
    def get_online_pred(self, features):
        ensemble_pred = np.zeros((len(features), self.output_dim))
        for client in self.active_clients:
            test_pred = client.pred(features)[0]
            for i in range(len(test_pred)):
                if test_pred[i] != -1:
                    ensemble_pred[i][test_pred[i]] += 1
        for i in range(len(test_pred)):
            ensemble_pred[i] = (ensemble_pred[i] + 0.01 / self.output_dim) / (sum(ensemble_pred[i]) + 0.01)
        return ensemble_pred
    
    def get_offline_pred(self, features):
        return self.offline_client.pred(features)[1]
    
    def save_states(self, path):
        with open(path + ".pickle", 'wb') as file:
            pickle.dump([self.clients, self.offline_client], file)

    def save_results(self, path, r, num_round):
        if not os.path.exists(path + ".csv"):
            with open(path + ".csv", 'w', newline='') as file:
                writer = csv.writer(file)
                if self.has_backdoor:
                    writer.writerow(["round", "active_ordinary_count", "active_anomaly_count", "avg_ordinary_tokens", "avg_anomaly_tokens",
                                    "test_loss_offline", "test_acc_offline",
                                    "test_macro_f1_offline", "test_macro_recall_offline", "test_macro_precision_offline",
                                    "test_micro_f1_offline", "test_micro_recall_offline", "test_micro_precision_offline",
                                    "test_weighted_f1_offline", "test_weighted_recall_offline", "test_weighted_precision_offline",
                                    "test_loss_online", "test_acc_online",
                                    "test_macro_f1_online", "test_macro_recall_online", "test_macro_precision_online",
                                    "test_micro_f1_online", "test_micro_recall_online", "test_micro_precision_online",
                                    "test_weighted_f1_online", "test_weighted_recall_online", "test_weighted_precision_online",
                                    "test_loss_offline_backdoor", "test_acc_offline_backdoor",
                                    "test_macro_f1_offline_backdoor", "test_macro_recall_offline_backdoor", "test_macro_precision_offline_backdoor",
                                    "test_micro_f1_offline_backdoor", "test_micro_recall_offline_backdoor", "test_micro_precision_offline_backdoor",
                                    "test_weighted_f1_offline_backdoor", "test_weighted_recall_offline_backdoor", "test_weighted_precision_offline_backdoor",
                                    "test_loss_online_backdoor", "test_acc_online_backdoor",
                                    "test_macro_f1_online_backdoor", "test_macro_recall_online_backdoor", "test_macro_precision_online_backdoor",
                                    "test_micro_f1_online_backdoor", "test_micro_recall_online_backdoor", "test_micro_precision_online_backdoor",
                                    "test_weighted_f1_online_backdoor", "test_weighted_recall_online_backdoor", "test_weighted_precision_online_backdoor"])
                else:
                    writer.writerow(["round", "active_ordinary_count", "active_anomaly_count", "avg_ordinary_tokens", "avg_anomaly_tokens",
                                    "test_loss_offline", "test_acc_offline",
                                    "test_macro_f1_offline", "test_macro_recall_offline", "test_macro_precision_offline",
                                    "test_micro_f1_offline", "test_micro_recall_offline", "test_micro_precision_offline",
                                    "test_weighted_f1_offline", "test_weighted_recall_offline", "test_weighted_precision_offline",
                                    "test_loss_online", "test_acc_online",
                                    "test_macro_f1_online", "test_macro_recall_online", "test_macro_precision_online",
                                    "test_micro_f1_online", "test_micro_recall_online", "test_micro_precision_online",
                                    "test_weighted_f1_online", "test_weighted_recall_online", "test_weighted_precision_online"])   
        active_ordinary_count = 0
        active_anomaly_count = 0
        avg_ordinary_tokens = 0
        avg_anomaly_tokens = 0
        for client in self.active_clients:
            if client.is_anomaly():
                active_anomaly_count += 1
                avg_anomaly_tokens += client.tokens
            else:
                active_ordinary_count += 1
                avg_ordinary_tokens += client.tokens
        avg_ordinary_tokens /= max(1, active_ordinary_count)
        avg_anomaly_tokens /= max(1, active_anomaly_count)
        offline_metrics = Utils.get_metrics(np.array(self.test_dataset[1]), self.get_offline_pred(self.test_dataset[0]))
        online_metrics = Utils.get_metrics(np.array(self.test_dataset[1]), self.get_online_pred(self.test_dataset[0]))
        if self.has_backdoor:
            offline_backdoor_metrics = Utils.get_metrics(np.array(self.backdoor_test_dataset[1]), self.get_offline_pred(self.backdoor_test_dataset[0]))
            online_backdoor_metrics = Utils.get_metrics(np.array(self.backdoor_test_dataset[1]), self.get_online_pred(self.backdoor_test_dataset[0]))
        else:
            offline_backdoor_metrics = []
            online_backdoor_metrics = []
        with open(path + ".csv", 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([r, active_ordinary_count, active_anomaly_count, avg_ordinary_tokens, avg_anomaly_tokens] + offline_metrics + online_metrics + offline_backdoor_metrics + online_backdoor_metrics)
        if self.has_backdoor:
            print(f"Save to {path + '.csv'}: {r} / {num_round} - num_generated_data: {len(self.public_dataset[0])}, "
                  f"test_loss_offline: {offline_metrics[0]}, test_acc_offline: {offline_metrics[1]}, "
                  f"test_loss_online: {online_metrics[0]}, test_acc_online: {online_metrics[1]}, "
                  f"test_loss_offline_backdoor: {offline_backdoor_metrics[0]}, test_acc_offline_backdoor: {offline_backdoor_metrics[1]}, "
                  f"test_loss_online_backdoor: {online_backdoor_metrics[0]}, test_acc_online_backdoor: {online_backdoor_metrics[1]}, "
                  f"active_clients: {len(self.active_clients)}({active_ordinary_count}/{active_anomaly_count}), avg_tokens: {avg_ordinary_tokens}/{avg_anomaly_tokens}")
        else:
            print(f"Save to {path + '.csv'}: {r} / {num_round} - num_generated_data: {len(self.public_dataset[0])}, "
                  f"test_loss_offline: {offline_metrics[0]}, test_acc_offline: {offline_metrics[1]}, "
                  f"test_loss_online: {online_metrics[0]}, test_acc_online: {online_metrics[1]}, "
                  f"active_clients: {len(self.active_clients)}({active_ordinary_count}/{active_anomaly_count}), avg_tokens: {avg_ordinary_tokens}/{avg_anomaly_tokens}")

    def run(self, num_round, num_epoch, required_data, batch_size, start_round=0):
        # local pretraining phase
        if start_round == 0:
            for client in tqdm.tqdm(self.active_clients, desc="training"):
                client.train(epochs=num_epoch)
            self.save_results(f"EDLSave_Results", 0, num_round)
            start_round += 1

        # distributed learning phase
        for r in range(start_round, num_round + 1):
            # required_data_per_round = -(-required_data * r * 2 // (num_round * (num_round + 1)))
            required_data_per_round = -(-required_data // num_round)
            self.public_dataset[0] = [None] * required_data_per_round
            self.public_dataset[1] = [np.zeros(self.output_dim) for _ in range(required_data_per_round)]
            pred_clients = [[[] for _ in range(self.output_dim)] for _ in range(required_data_per_round)]
            for batch in tqdm.tqdm(range(0, required_data_per_round, batch_size), desc="data generation"):
                num_data = min(batch_size, required_data_per_round - batch)
                # synthesis phase
                for i in range(num_data):
                    idx = random.randint(0, len(self.active_clients) - 1)
                    self.public_dataset[0][batch + i] = self.active_clients[idx].gen_data()
                
                # prediction phase
                for client in self.active_clients:
                    if client.tokens >= batch_size:
                        preds, _ = client.pred(self.public_dataset[0][batch : batch + num_data])
                        for i in range(num_data):
                            if preds[i] != -1:
                                client.tokens -= 1
                                self.public_dataset[1][batch + i][preds[i]] += 1
                                pred_clients[batch + i][preds[i]].append(client)
                # print(f"num_pred_list: {[sum(p) for p in self.public_dataset[1]]}")
                
                # profit phase
                for i in range(num_data):
                    diffs = Simulators.profit_percentage(self.public_dataset[1][batch + i])
                    num_participants = sum(self.public_dataset[1][batch + i])
                    # squared_num_participants = sum(np.array(self.public_dataset[1][batch + i]) ** 2)
                    # label = np.argmax(self.public_dataset[1][batch + i])
                    for j in range(self.output_dim):
                        self.public_dataset[1][batch + i][j] = (self.public_dataset[1][batch + i][j] + 0.01 / self.output_dim) / (num_participants + 0.01)
                        # self.public_dataset[1][batch + i][j] = (self.public_dataset[1][batch + i][j] + 0.01 / self.output_dim) ** 2 / (squared_num_participants + 0.01)
                        # self.public_dataset[1][batch + i][j] = 1 if j == label else 0
                        for client in pred_clients[batch + i][j]:
                            client.tokens += 1 + diffs[j] / 100
                drop_list = []
                for client in self.active_clients:
                    if client.tokens < batch_size:
                        self.dropped_clients.append(client)
                        drop_list.append(client)
                for drop in drop_list:
                    self.active_clients.remove(drop)
                    print(f"remove {'anomaly' if drop.is_anomaly() else 'ordinary'} client")
                # print(f"token_list: {[c.tokens for c in self.active_clients]}")
            
            # training phase
            self.offline_client.train(epochs=num_epoch, public_data=self.public_dataset)
            for client in tqdm.tqdm(self.active_clients, desc="training"):
                client.train(epochs=num_epoch, public_data=self.public_dataset)

            # evaluation phase
            self.save_results(f"EDLSave_Results", r, num_round)
            self.save_states(f"EDLSave")


if __name__ == "__main__":
    is_iid = True
    num_round = 200
    num_epoch = 10
    required_data = 200000
    start_tokens = required_data / 100.0
    batch_size = int(start_tokens) // 2
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
        clients = [Clients.OrdinaryClient(Models.nn_model(input_dim, input_dim * 2, output_dim, lr), [d[i * num_dataset_per_client : (i + 1) * num_dataset_per_client] for d in train_dataset], start_tokens) for i in range(num_ordinary_clients)]
        clients += [Clients.DirtyLabelClient(Models.nn_model(input_dim, input_dim * 2, output_dim, lr), [d[i * num_dataset_per_client : (i + 1) * num_dataset_per_client] for d in train_dataset], start_tokens) for i in range(num_ordinary_clients, num_clients)]
    else:
        train_dataset = Utils.features2cycled_dataset(train_features, cycle_cnt=num_clients // output_dim)
        test_dataset = Utils.features2cycled_dataset(test_features, cycle_cnt=num_clients // output_dim)
        clients = [Clients.OrdinaryClient(Models.nn_binary_model(input_dim, input_dim * 2, output_dim, lr), [d[i * num_dataset_per_client : (i + 1) * num_dataset_per_client] for d in train_dataset], start_tokens, target_label=i % output_dim) for i in range(num_ordinary_clients)]
        clients += [Clients.DirtyLabelClient(Models.nn_binary_model(input_dim, input_dim * 2, output_dim, lr), [d[i * num_dataset_per_client : (i + 1) * num_dataset_per_client] for d in train_dataset], start_tokens, target_label=i % output_dim) for i in range(num_ordinary_clients, num_clients)]
    offline_client = Clients.Client(Models.nn_model(input_dim, input_dim * 2, output_dim, lr), [[], []])
    # with open("/kaggle/input/edlpacket/EDLSave.pickle", 'rb') as file:
    #     clients, offline_client = pickle.load(file)
    proposed_method = EnsembleDistributedLearning(input_dim, output_dim, clients, offline_client, test_dataset)
    proposed_method.run(num_round, num_epoch, required_data, batch_size)