from abc import ABCMeta, abstractmethod

from frozendict import frozendict


class ResourceManager(metaclass=ABCMeta):
    def __init__(self):
        self.wv_filename = ""
        self.parsed_filename = ""

    @abstractmethod
    def write(self):
        """
        parse the raw file/files and write the data to disk
        :return:
        """
        pass

    @abstractmethod
    def read(self):
        """
        read the parsed file from disk
        :return:
        """
        pass

    def read_hashable(self):
        return frozendict(self.read())
