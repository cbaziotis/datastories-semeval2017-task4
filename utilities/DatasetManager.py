from abc import ABCMeta, abstractmethod


class DatasetManager(metaclass=ABCMeta):
    def __init__(self):
        self.processed_filename = ""
        self.parsed_filename = ""

    @abstractmethod
    def get_training(self):
        pass

    @abstractmethod
    def get_test(self):
        pass
