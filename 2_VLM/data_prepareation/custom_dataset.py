"""Register a training dataset with the bundled SWIFT integration.

Set MOUSEGPT_TRAIN_DATASET_ID to a dataset containing query, response, and images
fields. This registration hook is not needed for inference.
"""
import os

dataset_id = os.environ.get('MOUSEGPT_TRAIN_DATASET_ID')
if not dataset_id:
    raise RuntimeError(
        'Set MOUSEGPT_TRAIN_DATASET_ID to your training dataset identifier '
        'before using this registration hook. It is not required for inference.')

from datasets import Dataset as HfDataset
from modelscope import MsDataset

from swift.llm import get_dataset, register_dataset, get_dataset_from_repo
from swift.utils import get_logger

logger = get_logger()


class CustomDatasetName:
    stsb_en = 'mouse_gpt_dataset_4o'

def _preprocess_stsb(dataset: HfDataset) -> HfDataset:

    query = []
    response = []
    images = []
    for d in dataset:
        query.append(d['query'])
        response.append(d['response'])
        images.append(d['images'])
    return HfDataset.from_dict({'query': query, 'response': response, 'images': images})


register_dataset(CustomDatasetName.stsb_en, dataset_id, None, _preprocess_stsb, get_dataset_from_repo)


if __name__ == '__main__':
    # test dataset
    train_dataset, val_dataset = get_dataset([CustomDatasetName.stsb_en],
                                             check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
