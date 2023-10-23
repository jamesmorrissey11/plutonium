from abc import ABC, abstractmethod


class DataHandlingStrategy(ABC):
    dataset_name: str

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def write_text_to_file(self):
        pass
