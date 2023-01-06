import copy
from typing import Dict
from torch import nn

from src.federated.protocols import Trainer, Aggregator


# noinspection PyTypeChecker
class AVGAggregator(Aggregator):
    def aggregate(self, trainers_models_weight_dict: Dict[int, nn.ModuleDict], sample_size: Dict[int, int],
                  round_id: int) -> nn.ModuleDict:
        model_list = []
        training_num = 0

        for idx in trainers_models_weight_dict.keys():
            model_list.append((sample_size[idx], copy.deepcopy(trainers_models_weight_dict[idx])))
            training_num += sample_size[idx]

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
