import datasets
from pathlib import Path


datasets.config.DOWNLOADED_DATASETS_PATH = Path("/home/ana/data4/datasets/transformers_datasets")
datasets.load_dataset("nielsr/funsd-layoutlmv3")


if __name__ == '__main__':
    pass