import time

from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.utils import \
    read_csv_files, load_config, DATA_ENGINE_PANDAS


def read_raw_data(raw_data_path, data_engine):
    print('reading raw data...')
    return read_csv_files(raw_data_path, engine=data_engine)


def transform_data(data, transform_spec, data_engine):
    print("transforming data...")
    from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.data.data_transform \
        import DataTransformer
    data_transformer = DataTransformer(data, transform_spec, data_engine)
    return data_transformer.transform()


def split_data(data, data_splitting_rule, data_engine):
    print('splitting data...')
    from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.data.data_splitting \
        import DataSplitter
    data_splitter = DataSplitter(data, data_splitting_rule)
    train_data, test_data = data_splitter.split()
    return train_data, test_data


def post_transform(train_data, test_data, post_transform_spec, data_engine):
    print("transform pre-splitting data...")
    from cloudtik.runtime.ai.modeling.classical_ml.classification_and_regression.xgboost.modeling.data.post_transform \
        import PostTransformer
    data_transformer = PostTransformer(
        train_data, test_data, post_transform_spec, data_engine)
    train_data, test_data = data_transformer.transform()
    return train_data, test_data


def save_processed_data(train_data, test_data, output_file, data_engine):
    print('saving data...')
    if data_engine == DATA_ENGINE_PANDAS:
        import pandas as pd
    else:
        import modin.pandas as pd

    data = pd.concat([train_data, test_data])
    data.to_csv(output_file, index=False)
    print(f'data saved under the path {output_file}')


def process_data(raw_data_path, data_engine,
                 data_processing_config,
                 output_file):
    config = load_config(data_processing_config)
    transform_spec = config['data_transform']
    split_spec = config['data_splitting']
    post_transform_spec = config['post_transform']

    dp_start = time.time()
    start = time.time()
    data = read_raw_data(raw_data_path, data_engine)
    print("read data took %.1f seconds" % (time.time() - start))
    start = time.time()
    data = transform_data(data, transform_spec, data_engine)
    print("transform data took %.1f seconds" % (time.time() - start))
    start = time.time()
    train_data, test_data = split_data(data, split_spec, data_engine)
    data = None
    print("split data took %.1f seconds" % (time.time() - start))
    start = time.time()
    train_data, test_data = post_transform(train_data, test_data, post_transform_spec, data_engine)
    print("post transform data took %.1f seconds" % (time.time() - start))
    if output_file:
        start = time.time()
        save_processed_data(train_data, test_data, output_file, data_engine)
        print("save data took %.1f seconds" % (time.time() - start))
    print("data preprocessing took %.1f seconds" % (time.time() - dp_start))
    return train_data, test_data
