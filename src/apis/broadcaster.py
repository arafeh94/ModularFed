import typing
from abc import ABC, abstractmethod


class Subscriber(ABC):
    def __init__(self, only=None):
        self.only = only

    def force(self) -> []:
        return []

    @abstractmethod
    def map_events(self) -> typing.Dict[str, typing.Callable]:
        pass


class Broadcaster:
    def __init__(self):
        self.events = {}

    def broadcast(self, event_name: str, **kwargs):
        if event_name in self.events:
            for item in self.events[event_name]:
                item(kwargs)

    def register_event(self, event_name: str, action: callable):
        if event_name not in self.events:
            self.events[event_name] = []
        self.events[event_name].append(action)

    def add_subscriber(self, subscriber: Subscriber):
        events = subscriber.map_events()
        for event_name, call in events.items():
            if subscriber.only is not None and event_name not in subscriber.only:
                if event_name not in subscriber.force():
                    continue
            self.register_event(event_name, call)
