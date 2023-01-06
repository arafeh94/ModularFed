import torch
from torch import nn


class ParamException(Exception):
    def __init__(self, name, param):
        super(ParamException, self).__init__(f'{name} is missing parameter {param}')


def sgd(lr):
    """
    new instance of sgd optimizer
    :param lr: learn rate
    :return: new instance creator of sgd optim
    """
    return lambda model: torch.optim.SGD(model.parameters(), lr=lr)


def adam(lr, wd, amsgrad=True):
    """
    new instance of adam optimizer
    :param lr: learn rate
    :param wd: weight_decay
    :param amsgrad:
    :return: new instance creator of adam optim
    """
    return lambda model: torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd,
                                          amsgrad=amsgrad)


def optimizer(name, **kwargs):
    if name == 'adam':
        if 'lr' not in kwargs:
            ParamException(name, 'lr')
        if 'wd' not in kwargs:
            ParamException(name, 'wd')
        return adam(kwargs['lr'], kwargs['wd'])
    elif name == 'sgd':
        if 'lr' not in kwargs:
            ParamException(name, 'lr')
        return sgd(kwargs['lr'])
    Exception(f'unknown optimizer {name}')


def criterion(name, **kwargs):
    if name in ['cross_entropy_loss', 'cel']:
        return nn.CrossEntropyLoss()
    raise Exception(f'unknown criterion {name}')
