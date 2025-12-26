import random
import keras
import numpy as np

class Client():
    def __init__(self, model: keras.Model, private_data, tokens=0, target_label=None):
        self.model = model
        self.tokens = tokens
        self.target_label = target_label
        self.private_features = private_data[0]
        if self.target_label is None:
            self.private_labels = private_data[1]
        else:
            self.private_labels = [1 if np.argmax(label) == self.target_label else 0 for label in private_data[1]]

    def is_anomaly(self):
        pass

    def gen_data(self, sensitivity=1e-3, epsilon=5.0):
        data = self.private_features[random.randint(0, len(self.private_features) - 1)]
        noise = np.random.laplace(0, sensitivity / epsilon, len(self.private_features[0]))
        return data + noise
    
    def train(self, epochs=1, public_data=[[], []]):
        if self.target_label is None:
            public_features = public_data[0]
            public_labels = public_data[1]
            features = np.array(self.private_features + public_features)
            labels = np.array(self.private_labels + public_labels)
        else:
            features_pos = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 1] + [feature for feature, label in zip(*public_data) if label[self.target_label] > 0.5]
            features_neg = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 0] + [feature for feature, label in zip(*public_data) if label[self.target_label] <= 0.5]
            if len(features_pos) > len(features_neg):
                gen_neg_cnt = len(features_pos) - len(features_neg)
                gen_features_neg = []
                for _ in range(gen_neg_cnt):
                    rate = random.random() * 0.2 + 0.5
                    gen_features_neg.append(rate * self.private_features[random.randint(0, len(self.private_features) - 1)] + (1 - rate) * np.random.normal(0, 0.01, len(self.private_features[0])))
                features = np.array(features_pos + features_neg + gen_features_neg)
            else:
                features = np.array(features_pos + features_neg[:len(features_pos)])
            labels = np.array([1 for _ in range(len(features_pos))] + [0 for _ in range(len(features_pos))])
        self.model.fit(features, labels, batch_size=64, epochs=epochs, verbose=0)

    def pred(self, data):
        preds = self.model.predict(np.array(data), verbose=0)
        if self.target_label is None:
            return np.argmax(preds, axis=-1), preds
        else:
            return np.array([self.target_label if pred[0] > 0.5 else -1 for pred in preds]), preds

    def eval(self, test_data):
        loss, acc = self.model.evaluate(np.array(test_data[0]), np.array(test_data[1]), batch_size=64, verbose=0)
        return loss, acc

class OrdinaryClient(Client):
    def __init__(self, model:keras.Model, private_data, tokens=0, target_label=None):
        super().__init__(model, private_data, tokens, target_label)

    def is_anomaly(self):
        return False

class DirtyLabelClient(Client):
    def __init__(self, model:keras.Model, private_data, tokens=0, target_label=None):
        super().__init__(model, private_data, tokens, target_label)

    def is_anomaly(self):
        return True

    def train(self, epochs=1, public_data=[[], []]):
        if self.target_label is None:
            public_features = public_data[0]
            public_labels = public_data[1]
            classes = len(self.private_labels[0])
            features = np.array(self.private_features + public_features)
            labels = keras.utils.to_categorical(classes - 1 - np.argmax(self.private_labels + public_labels, axis=-1), num_classes=classes)
        else:
            features_pos = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 1] + [feature for feature, label in zip(*public_data) if label[self.target_label] > 0.5]
            features_neg = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 0] + [feature for feature, label in zip(*public_data) if label[self.target_label] <= 0.5]
            if len(features_pos) > len(features_neg):
                gen_neg_cnt = len(features_pos) - len(features_neg)
                gen_features_neg = []
                for _ in range(gen_neg_cnt):
                    rate = random.random() * 0.2 + 0.5
                    gen_features_neg.append(rate * self.private_features[random.randint(0, len(self.private_features) - 1)] + (1 - rate) * np.random.normal(0, 0.01, len(self.private_features[0])))
                features = np.array(features_pos + features_neg + gen_features_neg)
            else:
                features = np.array(features_pos + features_neg[:len(features_pos)])
            labels = 1 - np.array([1 for _ in range(len(features_pos))] + [0 for _ in range(len(features_pos))])
        self.model.fit(features, labels, batch_size=64, epochs=epochs, verbose=0)

class CleanLabelClient(Client):
    def __init__(self, model:keras.Model, private_data, tokens=0, target_label=None):
        super().__init__(model, private_data, tokens, target_label)

    def is_anomaly(self):
        return True
    
    def train(self, epochs=1, public_data=[[], []]):
        if self.target_label is None:
            public_features = public_data[0]
            public_labels = public_data[1]
            features = np.array(self.private_features + public_features)
            labels = np.array(self.private_labels + public_labels)
        else:
            features_pos = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 1] + [feature for feature, label in zip(*public_data) if label[self.target_label] > 0.5]
            features_neg = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 0] + [feature for feature, label in zip(*public_data) if label[self.target_label] <= 0.5]
            if len(features_pos) > len(features_neg):
                gen_neg_cnt = len(features_pos) - len(features_neg)
                gen_features_neg = []
                for _ in range(gen_neg_cnt):
                    rate = random.random() * 0.2 + 0.5
                    gen_features_neg.append(rate * self.private_features[random.randint(0, len(self.private_features) - 1)] + (1 - rate) * np.random.normal(0, 0.01, len(self.private_features[0])))
                features = np.array(features_pos + features_neg + gen_features_neg)
            else:
                features = np.array(features_pos + features_neg[:len(features_pos)])
            labels = np.array([1 for _ in range(len(features_pos))] + [0 for _ in range(len(features_pos))])
        features = 0.5 * features + 0.5 * np.flip(features, axis=0)
        self.model.fit(features, labels, batch_size=64, epochs=epochs, verbose=0)

class BackdoorClient(Client):
    def __init__(self, model:keras.Model, private_data, tokens=0, target_label=None):
        super().__init__(model, private_data, tokens, target_label)
        mask = np.ones(len(self.private_features[0]))
        mask[:len(self.private_features[0]) // 16] = 0
        self.private_features += [feature * mask + 0.5 * (1 - mask) for feature in self.private_features]
        if self.target_label is None:
            classes = len(self.private_labels[0])
            self.private_labels += list(keras.utils.to_categorical(classes - 1 - np.argmax(self.private_labels, axis=-1), num_classes=classes))
        else:
            self.private_labels += [0 if np.argmax(label) == target_label else 1 for label in self.private_labels]

    def is_anomaly(self):
        return True

class ModelManipulationClient(Client):
    def __init__(self, model:keras.Model, private_data, tokens=0, target_label=None):
        super().__init__(model, private_data, tokens, target_label)

    def is_anomaly(self):
        return True
    
    def train(self, epochs=1, public_data=[[], []]):
        if self.target_label is None:
            public_features = public_data[0]
            public_labels = public_data[1]
            features = np.array(self.private_features + public_features)
            labels = np.array(self.private_labels + public_labels)
        else:
            features_pos = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 1] + [feature for feature, label in zip(*public_data) if label[self.target_label] > 0.5]
            features_neg = [feature for feature, label in zip(self.private_features, self.private_labels) if label == 0] + [feature for feature, label in zip(*public_data) if label[self.target_label] <= 0.5]
            if len(features_pos) > len(features_neg):
                gen_neg_cnt = len(features_pos) - len(features_neg)
                gen_features_neg = []
                for _ in range(gen_neg_cnt):
                    rate = random.random() * 0.2 + 0.5
                    gen_features_neg.append(rate * self.private_features[random.randint(0, len(self.private_features) - 1)] + (1 - rate) * np.random.normal(0, 0.01, len(self.private_features[0])))
                features = np.array(features_pos + features_neg + gen_features_neg)
            else:
                features = np.array(features_pos + features_neg[:len(features_pos)])
            labels = np.array([1 for _ in range(len(features_pos))] + [0 for _ in range(len(features_pos))])
        prev_weights = self.model.get_weights()
        self.model.fit(features, labels, batch_size=64, epochs=epochs, verbose=0)
        modify_weights = [prev_layer_weights + (layer_weights - prev_layer_weights) * -100 for prev_layer_weights, layer_weights in zip(prev_weights, self.model.get_weights())]
        self.model.set_weights(modify_weights)