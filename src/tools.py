import copy
import math
import typing
from datetime import datetime, timedelta

import numpy as np
import torch
import tqdm
from sklearn import decomposition
from torch import nn
import logging
from src.data.data_container import DataContainer

logger = logging.getLogger('tools')


def dict_select(idx, dict_ref):
    new_dict = {}
    for i in idx:
        new_dict[i] = dict_ref[i]
    return new_dict


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params


def flatten_weights(weights, compress=False):
    weight_vecs = []
    for _, weight in weights.items():
        weight_vecs.extend(weight.flatten().tolist())
    if compress:
        return compress_weights(np.array(weight_vecs))
    return np.array(weight_vecs)


def compress_weights(flattened_weights):
    weights = flattened_weights.reshape(10, -1)
    pca = decomposition.PCA(n_components=4)
    pca.fit(weights)
    weights = pca.transform(weights)
    return weights.flatten()


def timed_func(seconds, callable: typing.Callable):
    stop = datetime.now() + timedelta(seconds=seconds)
    while datetime.now() < stop:
        callable()


def train(model, train_data, epochs=10, lr=0.1):
    torch.cuda.empty_cache()
    # change to train mode
    device = torch.device('cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    epoch_loss = []
    for epoch in tqdm.tqdm(range(epochs), 'training'):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            log_probs = model(x)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        if len(batch_loss) > 0:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    weights = model.cpu().state_dict()
    return weights


def aggregate(models_dict: dict, sample_dict: dict):
    model_list = []
    training_num = 0

    for idx in models_dict.keys():
        model_list.append((sample_dict[idx], copy.deepcopy(models_dict[idx])))
        training_num += sample_dict[idx]

    # logging.info("################aggregate: %d" % len(model_list))
    (num0, averaged_params) = model_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            w = local_sample_number / training_num
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w

    return averaged_params


def infer(model, test_data):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            pred = model(x)
            loss = criterion(pred, target)
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            test_acc += correct.item()
            test_loss += loss.item() * target.size(0)
            test_total += target.size(0)

    return test_acc / test_total, test_loss / test_total


def load(model, stats):
    model.load_state_dict(stats)


def influence_ecl(aggregated, model):
    all = []
    for key in aggregated.keys():
        l2_norm = torch.dist(aggregated[key], model[key], 2)
        val = l2_norm.numpy().min()
        all.append(val)
    return math.fsum(all) / len(all)


def influence_cos(model1, model2, aggregated):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    center = torch.flatten(aggregated["linear.weight"])
    p1 = torch.flatten(model1["linear.weight"])
    p2 = torch.flatten(model2["linear.weight"])
    p1 = torch.subtract(center, p1)
    p2 = torch.subtract(center, p2)
    return cos(p1, p2).numpy().min()


def influence_cos2(aggregated, model):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    x1 = torch.flatten(aggregated["linear.weight"])
    x2 = torch.flatten(model["linear.weight"])
    return cos(x1, x2).numpy().min()


def normalize(arr, z1=False):
    if z1:
        return (arr - min(arr)) / (max(arr) - min(arr))
    return np.array(arr) / math.fsum(arr)


class Dict:
    @staticmethod
    def select(idx, dict_ref):
        new_dict = {}
        for i in idx:
            new_dict[i] = dict_ref[i]
        return new_dict

    @staticmethod
    def but(keys, dict_ref):
        new_dict = {}
        for item, val in dict_ref.items():
            if item not in keys:
                new_dict[item] = val
        return new_dict

    @staticmethod
    def concat(first, second):
        new_dict = {}
        for item, val in first.items():
            new_dict[item] = val
        for item, val in second.items():
            new_dict[item] = val
        return new_dict


def detail(client_data: typing.Union[typing.Dict[int, DataContainer], DataContainer], selection=None,
           display: typing.Callable = None):
    if display is None:
        display = lambda x: logger.info(x)
    if isinstance(client_data, DataContainer):
        client_data = {0: client_data}
    display("<--clients_labels-->")
    for client_id, data in client_data.items():
        if selection is not None:
            if client_id not in selection:
                continue
        uniques = np.unique(data.y)
        display(f"client_id: {client_id} --size: {len(data.y)} --num_labels: {len(uniques)} --unique_labels:{uniques}")
        for unique in uniques:
            unique_count = 0
            for item in data.y:
                if item == unique:
                    unique_count += 1
            percentage = unique_count / len(data.y) * 100
            percentage = int(percentage)
            display(f"labels_{unique}= {percentage}% - {unique_count}")


def compress(weights, output_dim, n_components):
    weights = weights.flatten().reshape(output_dim, -1)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(weights)
    weights = pca.transform(weights)
    return weights.flatten()
