import copy
import typing
import numpy as np
import torch
import tqdm
from torch import nn
import logging
from src.data.data_container import DataContainer

logger = logging.getLogger('tools')


def train(model, train_data, epochs=10, lr=0.1):
    torch.cuda.empty_cache()
    # change to train mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        if idx not in sample_dict:
            sample_dict[idx] = 1
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


def infer(model, test_data, transformer=None):
    model.eval()
    test_loss = test_acc = test_total = 0.
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            x = x.to('cuda')
            target = target.to('cuda')
            if transformer:
                x, target = transformer(x, target)
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
