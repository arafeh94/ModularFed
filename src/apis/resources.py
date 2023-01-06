import os
from abc import abstractmethod

import psutil
from matplotlib import pyplot as plt

from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning
from apscheduler.schedulers.background import BackgroundScheduler


class Tracker:
    def __init__(self, title, frequency=1):
        self.delay = 1 / frequency
        self.tracks = []
        self.title = title
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self._tracker, 'interval', seconds=self.delay)

    @abstractmethod
    def supplier(self):
        pass

    def _tracker(self):
        dt = self.supplier()
        self.tracks.append(dt)

    def start(self):
        self.scheduler.start()

    def stop(self):
        self.scheduler.remove_all_jobs()

    def plot(self, title=None):
        plt.title(title or self.title)
        plt.xlabel('Time Slot')
        plt.ylabel('Value')
        plt.plot(self.tracks)
        plt.show()

    class TrackSubscriber(FederatedSubscriber):
        def __init__(self, tracker: 'Tracker'):
            super().__init__()
            self.tracker = tracker

        def on_federated_started(self, params):
            self.tracker.start()

        def on_federated_ended(self, params):
            self.tracker.stop()
            self.tracker.plot(self.tracker.title)

    def attach(self, federated: FederatedLearning):
        federated.add_subscriber(self.TrackSubscriber(self))


class RamTracker(Tracker):
    def __init__(self, delay=1):
        super().__init__('RAM Usage', delay)
        self.process = psutil.Process(os.getpid())

    def supplier(self):
        return round(self.process.memory_info().rss / 1e+6)


class CPUTracker(Tracker):
    def __init__(self, delay=1):
        super().__init__('CPU Usage', delay)
        self.process = psutil.Process(os.getpid())

    def supplier(self):
        return round(self.process.cpu_percent())
