# Downloader
Utility for downloading and managing AI datasets and models.

## Datasets

### Supported Sources & File Types

| Source | Info |
|----------|-----------|
| TensorFlow Datasets | [https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets) |
| Torchvision | [https://pytorch.org/vision/stable/datasets.html](https://pytorch.org/vision/stable/datasets.html) |
| Hugging Face | [https://huggingface.co/docs/datasets/index](https://huggingface.co/docs/datasets/index) |
| Generic Web URL | Publicly downloadable files: `.zip`, `.gz`, `.bz2`, `.txt`, `.csv`, `.png`, `.jpg`, etc. |

### Usage

Dataset source example:
```
from downloader.datasets import DataDownloader

downloader = DataDownloader('tf_flowers', dataset_dir='/home/user/datasets', source='tensorflow_datasets')
downloader.download(split='train')
```

URL example:
```
from downloader.datasets import DataDownloader

downloader = DataDownloader('my_dataset', dataset_dir='/home/user/datasets', url='http://<domain>/<filename>.zip')
downloader.download()
```

## Models

### Supported Model Hubs

| Source | Info |
|----------|-----------|
| TensorFlow Hub | [https://www.tensorflow.org/hub](https://www.tensorflow.org/hub) |
| Torchvision | [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html) |
| Hugging Face | [https://huggingface.co/models](https://huggingface.co/models) (AutoModelForSequenceClassification type) |

### Usage

Example:
```
from downloader.models import ModelDownloader

# Hugging Face
downloader = ModelDownloader('bert-large-uncased', source='hugging_face', num_labels=2)
downloader.download()

# Torchvision
downloader = ModelDownloader('resnet34', hub='torchvision')
downloader.download()
```
