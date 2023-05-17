# Downloader

An easy-to-use, unified tool for downloading and managing AI datasets and models.

## Datasets

### Supported Catalogs & File Types

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
downloader = ModelDownloader('bert-large-uncased', hub='hugging_face', num_labels=2)
downloader.download()

# Torchvision
downloader = ModelDownloader('resnet34', hub='torchvision')
downloader.download()
```

## Build and Install

To install the downloader, follow [The setup instructions for Intel Transfer Learning Tool](/README.md#build-and-install). The downloader is currently
packaged alongside the Intel Transfer Learning Tool and uses its requirements.txt files, but the tools can be separated at some future time. The
downloader's dependencies are tracked in [requirements.txt](requirements.txt).

## Testing
With an activated environment that has the dependencies for the downloader and `pytest` in it, run this command from
the root repository directory:

```
py.test -s downloader/tests
```
