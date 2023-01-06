import argparse
import typing


class FederatedArgs:
    def __init__(self, defaults: typing.Union[typing.Dict, None] = None):
        """
        defaults keys:
        epoch, batch, round, shard, dataset, clients_ratio, learn_rate, tag, min, max, clients
        example of execution:
        fed.py -e 25 -b 50 -r 100 -s 2 -d mnist -cr 0.1 -lr 0.1 -t mnist10 -mn 600 -mx 600 -cln 100
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epoch', type=int, help='epochs count',
                            default=defaults['epoch'] if defaults else 2)
        parser.add_argument('-b', '--batch', type=int, help='batch count',
                            default=defaults['batch'] if defaults else 50)
        parser.add_argument('-r', '--round', type=int, help='number of rounds',
                            default=defaults['round'] if defaults else 10)
        parser.add_argument('-s', '--distributor', type=str, help='shard count max 10',
                            default=defaults['distributor'] if defaults else 'none')
        parser.add_argument('-d', '--dataset', type=str, help='dataset mnist or cifar10',
                            default=defaults['dataset'] if defaults else 'mnist')
        parser.add_argument('-cr', '--clients_ratio', type=float, help='selected client percentage for fl',
                            default=defaults['clients_ratio'] if defaults else 0.1)
        parser.add_argument('-lr', '--learn_rate', type=float, help='learn rate',
                            default=defaults['learn_rate'] if defaults else 0.1)
        parser.add_argument('-t', '--tag', type=str, help='tag to save the results',
                            default=defaults['tag'] if defaults else 'def_tag')
        parser.add_argument('-mn', '--min', type=int, help='minimum number of data',
                            default=defaults['min'] if defaults else 600)
        parser.add_argument('-mx', '--max', type=int, help='maximum number of data',
                            default=defaults['max'] if defaults else 600)
        parser.add_argument('-cln', '--clients_number', type=int, help='number of participating clients',
                            default=defaults['clients'] if defaults else 100)
        args = parser.parse_args()
        self._validate(args)
        self.epoch = args.epoch
        self.batch = args.batch
        self.round = args.round
        self.distributor = args.distributor
        self.dataset = args.dataset
        self.clients_ratio = args.clients_ratio
        self.min = args.min
        self.max = args.max
        self.clients = args.clients_number
        self.learn_rate = args.learn_rate
        self.tag = args.tag
        self.timestamp = defaults['timestamp'] if 'timestamp' in defaults else ''
        self.model = defaults['mode'] if 'model' in defaults else ''

    def _validate(self, args):
        for key, item in args.__dict__.items():
            if item is None:
                Exception(key + ' value is missing')

    def __repr__(self):
        return f'{self.tag}_e{self.epoch}_b{self.batch}_r{self.round}_dis#{self.distributor}' \
               f'_{self.dataset}_cr{str(self.clients_ratio).replace(".", "")}' \
               f'_lr{str(self.learn_rate)}{self.timestamp}'.replace('cr1', 'cr10')
