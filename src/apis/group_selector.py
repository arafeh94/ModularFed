from sklearn.cluster import KMeans

from src.apis import utils


def cluster(model_stats, cluster_size):
    weights = []
    client_ids = []
    clustered = {}
    for client_id, stats in model_stats.items():
        client_ids.append(client_id)
        weights.append(utils.flatten_weights(stats))
    kmeans = KMeans(n_clusters=cluster_size).fit(weights)
    for i, label in enumerate(kmeans.labels_):
        clustered[client_ids[i]] = label
    return clustered


class GroupSelector:
    def __init__(self, clients_model_dict: dict):
        """
        @param clients_model_dict dictionary of user id, and the trained model of this user
        """
        self.id_label_dict = clients_model_dict
        self.used_clusters = []
        self.used_models = []

    def reset(self):
        self.used_clusters = []
        self.used_models = []

    def select(self, model_id):
        if model_id in self.used_models:
            return False
        self.used_clusters.append(self.id_label_dict[model_id])
        self.used_models.append(model_id)
        return model_id

    def random(self, size):
        selected_idx = []
        while len(selected_idx) < size:
            available = self.list()
            if len(available) == 0:
                self.reset()
            else:
                selected_idx.append(self.select(available[0]))
        return selected_idx

    def list(self):
        if len(self.used_models) == len(self.id_label_dict):
            return []
        model_ids = []
        for model_id, label in self.id_label_dict.items():
            if label not in self.used_clusters and model_id not in self.used_models:
                model_ids.append(model_id)
        if len(model_ids) == 0:
            self.used_clusters = []
            return self.list()
        return model_ids

    def __len__(self):
        return len(self.id_label_dict.keys())

    @staticmethod
    def build(model_stats, cluster_size):
        clustered = cluster(model_stats, cluster_size)
        return GroupSelector(clustered)
