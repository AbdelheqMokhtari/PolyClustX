from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X):
        pass
