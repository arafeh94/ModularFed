import os
import psutil
import sys
from abc import abstractmethod

from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

from src.apis import utils
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


class BasePlotter(FederatedSubscriber):
    def __init__(self, plot_ratio=1, show_plot=True, save_prefix=None, plot_title=None):
        super().__init__()
        self.plot_ratio = plot_ratio if plot_ratio else sys.maxsize
        self.show_plot = show_plot
        self.save_prefix = save_prefix
        self.plot_title = plot_title

    def on_round_end(self, params):
        context = params['context']
        if context.round_id > 0 and context.round_id % self.plot_ratio == 0:
            plot = self.round_plot(params)
            self.execute(plot, context)

    def on_federated_ended(self, params):
        context = params['context']
        plot = self.final_plot(params)
        self.execute(plot, context)

    def execute(self, plot, context):
        if not plot:
            return
        if self.save_prefix:
            file_name = f'{self.save_prefix}_{self.save_file_name(context)}.png'
            plot.savefig(file_name)
        if self.show_plot:
            plot.show()

    @abstractmethod
    def round_plot(self, params):
        pass

    @abstractmethod
    def final_plot(self, params):
        pass

    @abstractmethod
    def save_file_name(self, context: FederatedLearning.Context):
        pass


class LocalAccuracy(BasePlotter):
    def final_plot(self, params):
        return None

    def round_plot(self, params) -> plt:
        plt.bar(params['local_acc'].keys(), params['local_acc'].values())
        plt.title('Local Accuracy')
        return plt

    def save_file_name(self, context: FederatedLearning.Context):
        return f'local_acc_{context.round_id}.png'


class RoundAccuracy(BasePlotter):
    def round_plot(self, params):
        acc = []
        history = params['context'].history
        for round_id, item in history.items():
            acc.append(item['acc'])
        plt.plot(acc)
        plt.title(self.plot_title or 'Round Accuracy')
        return plt

    def final_plot(self, params):
        return self.round_plot(params)

    def save_file_name(self, context: FederatedLearning.Context):
        return f'acc_{context.round_id}'


class LocalLoss(BasePlotter):
    def round_plot(self, params) -> plt:
        plt.bar(params['local_loss'].keys(), params['local_loss'].values())
        plt.title(self.plot_title or 'Local Loss')
        return plt

    def save_file_name(self, context: FederatedLearning.Context):
        return f'local_loss_{context.round_id}.png'

    def final_plot(self, params):
        return None


class FinalAccuracyPlot(RoundAccuracy):

    def __init__(self):
        super().__init__(0, True, None, 'Final Acc')

    def final_plot(self, params):
        return super().round_plot(params)


class RoundLoss(BasePlotter):
    def round_plot(self, params):
        acc = []
        history = params['context'].history
        for round_id, item in history.items():
            acc.append(item['loss'])
        plt.title(self.plot_title or 'Round Loss')
        plt.plot(acc)
        return plt

    def final_plot(self, params):
        return self.round_plot(params)

    def save_file_name(self, context: FederatedLearning.Context):
        return f'loss_{context.round_id}'


class CPUUsage(BasePlotter):
    def __init__(self):
        super().__init__()
        self.process = psutil.Process(os.getpid())
        self.usage = []

    def round_plot(self, params):
        self.usage.append(self.process.cpu_percent())
        return None

    def final_plot(self, params):
        plt.title(self.plot_title or 'CPU Usage')
        plt.plot(self.usage)
        return plt

    def save_file_name(self, context: FederatedLearning.Context):
        return f'ram_{context.round_id}'


class EMDWeightDivergence(BasePlotter):
    def __init__(self, plot_ratio=1, show_plot=True):
        super().__init__(plot_ratio=plot_ratio, show_plot=show_plot)
        self.global_weights = None
        self.trainers_weights = None

    def on_training_end(self, params):
        self.trainers_weights = params['trainers_weights']

    def on_aggregation_end(self, params):
        self.global_weights = params['global_weights']

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        avg_weight_divergence = self._get_average_weight_divergence(self.global_weights, self.trainers_weights)
        context.store(wd=avg_weight_divergence)
        super(EMDWeightDivergence, self).on_round_end(params)

    def round_plot(self, params):
        wd = []
        history = params['context'].history
        for round_id, item in history.items():
            wd.append(item['wd'])
        figure = plt.figure(1)
        plot = figure.add_subplot()
        plot.plot(wd)
        plot.set_title(self.plot_title or 'EMD Weight Divergence')
        return figure

    def final_plot(self, params):
        return self.round_plot(params)

    def save_file_name(self, context: FederatedLearning.Context):
        return f'wd_{context.round_id}'

    def _get_average_weight_divergence(self, global_model_dict, trainers_weights):
        all_results = []
        flattened_global_weights = utils.flatten_weights(global_model_dict)
        for trained_id, trainers_weight in trainers_weights.items():
            flattened_trainer_weights = utils.flatten_weights(trainers_weight)
            result = wasserstein_distance(flattened_global_weights, flattened_trainer_weights)
            all_results.append(result)
        all_results = sum(all_results) / len(all_results)
        return all_results
