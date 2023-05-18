import os
from typing import List, Optional

import pandas as pd
from datasets.arrow_dataset import Dataset

from cloudtik.runtime.ai.modeling.transfer_learning.common.pytorch.hugging_face.dataset import HuggingFaceDataset
from ....text_classification.text_classification_dataset import \
    TextClassificationDataset


class UserTextClassificationDataset(TextClassificationDataset, HuggingFaceDataset):
    """
    A user custom text classification dataset that can be used with Transformer models.
    """

    def __init__(
        self,
        dataset_dir,
        dataset_name: Optional[str],
        csv_file_name: str,
        class_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        label_map_func: Optional[callable] = None,
        label_col: Optional[int] = 0,
        delimiter: Optional[str] = ",",
        header: Optional[bool] = False,
        select_cols: Optional[List[int]] = None,
        exclude_cols: Optional[List[int]] = None,
        shuffle_files: Optional[bool] = True,
        num_workers: Optional[int] = 0,
    ):
        """
        A custom text classification dataset that can be used with Transformer models.
        Note that this dataset class expects a .csv file with two columns where the first column is the label and
        the second column is the text/sentence to classify.

        For example, a comma separated value file will look similar to the snippet below:

        .. code-block:: text

            class_a,<text>
            class_b,<text>
            class_a,<text>
            ...

        If the .csv files has more columns, the select_cols or exclude_cols parameters can be used to filter out which
        columns will be parsed.

        Args:
            dataset_dir (str): Directory containing the dataset
            dataset_name (str): Name of the dataset. If no dataset name is given, the dataset_dir folder name
                will be used as the dataset name.
            csv_file_name (str): Name of the file to load from the dataset directory
            class_names (list(str)): optional; List of ordered class names. If None, class_names are inferred from
                label_col column
            column_names (list(str)): optional; List of column names. If given, there must be exactly one value as
                "label" in the position corresponding to the 'label_col' argument. If None, column names are assigned
                as "label" for the label_col column and "text_1", "text_2", ... for the rest of the columns.
            label_map_func (function): optional; Maps the label_map_func across the label column of the dataset to
                apply a transform to the elements. For example, if the .csv file has string class labels
                instead of numerical values, you can provide a function that maps the string to a numerical
                value or specify the index of the label column to apply a default label_map_func which assigns an
                integer for every unique class label, starting with 0.
            label_col (int): optional; Column index of the dataset to use as label column. Defaults to "0"
            delimiter (str): String character that separates the text in each row. Defaults to ","
            header (bool): optional; Boolean indicating whether or not the csv file has a header line that should be
                skipped. Defaults to False.
            select_cols (list): optional; Specify a list of sorted indices for columns from the dataset file(s) that
                should be parsed. Defaults to parsing all columns. At most one of select_cols and exclude_cols can
                be specified.
            exclude_cols (list): optional; Specify a list of sorted indices for columns from the dataset file(s) that
                should be excluded from parsing. Defaults to parsing all columns. At most one of select_cols and
                exclude_cols can be specified.
            shuffle_files (bool): optional; Whether to shuffle the data. Defaults to True.
            num_workers (int): Number of workers to pass into a DataLoader.

        Raises:
            FileNotFoundError if the csv file is not found in the dataset directory
            TypeError if label_map_func is not callable
            ValueError if class_names list is empty
            ValueError if column_names list does not contain the value 'label'
            ValueError if index of 'label' in column_names and label_col mismatch
            ValueError if the values of column_names are not strings.
            ValueError if column_names contains more than one value as 'label'

        """
        # Sanity checks
        dataset_file = os.path.join(dataset_dir, csv_file_name)
        if not os.path.exists(dataset_file):
            raise FileNotFoundError("The dataset file ({}) does not exist".format(dataset_file))

        if isinstance(class_names, list) and len(class_names) == 0:
            raise ValueError("The class_names list cannot be empty.")

        if label_map_func and not callable(label_map_func):
            raise TypeError("The label_map_func is expected to be a function, but found a {}", type(label_map_func))

        # The dataset name is only used for informational purposes. Default to use the file name without extension.
        if not dataset_name:
            dataset_name = os.path.splitext(csv_file_name)[0]

        if column_names:
            if 'label' not in column_names:
                raise ValueError("The column_names list must contain one value as 'label'")
            if column_names.count('label') > 1:
                raise ValueError("There must be exactly one value as 'label' in column_names.")
            if not all(isinstance(c, str) for c in column_names):
                raise ValueError("All column names must be strings.")
            if column_names.index('label') != label_col:
                raise ValueError("The label_col index ({}) does not match with column_names {}."
                                 "Either specify label_col argument (or) make the first value "
                                 "in your column_names as 'label'".format(label_col, column_names))

        TextClassificationDataset.__init__(self, dataset_dir, dataset_name)

        print("WARNING: Using column {} as label column. To change this behavior, \
               specify the label_col argument".format(label_col))
        if header:
            dataset_df = pd.read_csv(dataset_file, delimiter=delimiter, encoding='utf-8', dtype=str, names=column_names,
                                     header=0)
        else:
            dataset_df = pd.read_csv(dataset_file, delimiter=delimiter, encoding='utf-8', dtype=str, names=column_names,
                                     header=None)
            if not column_names:
                column_names = {i: 'label' if i == label_col else f'text_{i}' for i in dataset_df.columns}
                dataset_df.rename(column_names, axis=1, inplace=True)

        if select_cols and not exclude_cols:
            dataset_df = dataset_df[dataset_df.columns[select_cols]]
        elif exclude_cols and not select_cols:
            dataset_df = dataset_df.drop(dataset_df.columns[exclude_cols], axis=1)
        elif select_cols and exclude_cols:
            if not set(select_cols).isdisjoint(exclude_cols):
                raise ValueError("select_cols and exclude_cols lists are ambiguous. \
                                  Please make sure they are disjoint")
            dataset_df = dataset_df.drop(dataset_df.columns[exclude_cols], axis=1)
            dataset_df = dataset_df[dataset_df.columns[select_cols]]

        if not class_names:
            class_names = dataset_df.iloc[:, label_col].unique()

        if not label_map_func:
            label_str_dict = {label_name: idx for idx, label_name in enumerate(class_names)}

            def label_map_func(x):
                return label_str_dict[x]

        dataset_df.iloc[:, label_col] = dataset_df.iloc[:, label_col].map(label_map_func)

        self._dataset = Dataset.from_pandas(dataset_df)

        self._info = {
            "name": dataset_name,
            "dataset_dir": dataset_dir,
            "file_name": csv_file_name,
            "delimiter": delimiter,
            "header": header,
            "select_cols": select_cols,
            "exclude_cols": exclude_cols,
            'class_names': class_names
        }

        self._class_names = class_names
        self._validation_type = None
        self._preprocessed = {}
        self._shuffle = shuffle_files
        self._num_workers = num_workers

    @property
    def dataset(self):
        return self._dataset

    @property
    def class_names(self):
        return self._class_names

    @property
    def info(self):
        return {'dataset_info': self._info, 'preprocessing_info': self._preprocessed}
