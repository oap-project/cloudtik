# Copyright 2017 onwards, fast.ai, Inc.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import datetime
import h5py
import io
import os
import sys
import subprocess
from time import time
import math
from distutils.version import LooseVersion

parser = argparse.ArgumentParser(description='Spark Keras Rossmann Run Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--training-master',
                    help='spark cluster to use for training. If set to None, uses current default cluster. Cluster'
                         'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
                         'supplying `-c <NUM_GPUS>` in Spark Standalone mode. Example: spark://hostname:7077')
parser.add_argument('--num-proc', type=int, default=4,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--learning-rate', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--trials', type=int, default=2,
                    help='number of trails to parameter tuning (default: 2)')
parser.add_argument('--sample-rate', type=float,
                    help='desired sampling rate. Useful to set to low number (e.g. 0.01) to make sure that '
                         'end-to-end process works')
parser.add_argument('--data-dir',
                    help='location of data on local filesystem (prefixed with file://) or on distributed storage')
parser.add_argument('--work-dir',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--fsdir',
                    help='the file system dir (default: None)')
parser.add_argument('--local-submission-csv', default='submission.csv',
                    help='output submission predictions CSV on local filesystem (without file:// prefix)')


args = parser.parse_args()
fsdir = args.fsdir


# CloudTik cluster preparation or information
from cloudtik.runtime.spark.api import ThisSparkCluster
from cloudtik.runtime.ml.api import ThisMLCluster

cluster = ThisSparkCluster()

# Scale the cluster as need
# cluster.scale(workers=1)

# Wait for all cluster workers to be ready
cluster.wait_for_ready(min_workers=1)

# Total worker cores
cluster_info = cluster.get_info()
total_workers = cluster_info.get("total-workers")
if not total_workers:
    total_workers = 1

default_storage = cluster.get_default_storage()
if not fsdir:
    fsdir = default_storage.get("default.storage.uri") if default_storage else None
    if not fsdir:
        print("Must specify storage filesystem dir using -f.")
        sys.exit(1)

ml_cluster = ThisMLCluster()
mlflow_url = ml_cluster.get_services()["mlflow"]["url"]


# Checkpoint utilities
CHECKPOINT_HOME = "/tmp/ml/checkpoints"


def get_checkpoint_file(log_dir, file_id):
    return os.path.join(log_dir, 'checkpoint-{file_id}'.format(file_id=file_id))


def create_log_dir(experiment_name):
    log_dir = os.path.join(CHECKPOINT_HOME, str(time()), experiment_name)
    os.makedirs(log_dir)
    return log_dir


# HDFS driver to use with Petastorm.
PETASTORM_HDFS_DRIVER = 'libhdfs'


print('================')
print('Data preparation')
print('================')

# Download Rossmann dataset and upload to storage
if args.data_dir:
    data_dir = args.data_dir
else:
    # Download
    data_dir = fsdir + '/tmp/rossmann'

    data_url = 'http://files.fast.ai/part2/lesson14/rossmann.tgz'
    rossmann_data_file = os.path.join('/tmp', 'rossmann.tgz')
    temp_rossmann_data_path = '/tmp/rossmann'
    if not os.path.exists(rossmann_data_file):
        subprocess.check_output(['wget', data_url, '-O', rossmann_data_file])
        subprocess.check_output(['mkdir', temp_rossmann_data_path])
        subprocess.check_output(['tar', '--extract', '--file', rossmann_data_file, '--directory', temp_rossmann_data_path])

    # Upload to the default distributed storage
    os.system("hadoop fs -mkdir /tmp")
    os.system("hadoop fs -put /tmp/rossmann/ /tmp")


from pyspark import SparkConf, Row
from pyspark.sql import SparkSession
import pyspark.sql.types as T
import pyspark.sql.functions as F
import pyarrow as pa

# Create Spark session for data preparation.
spark_conf = SparkConf().setAppName('spark-keras-rossmann').set('spark.sql.shuffle.partitions', '16')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
conf = spark.conf

train_csv = spark.read.csv('%s/train.csv' % data_dir, header=True)
test_csv = spark.read.csv('%s/test.csv' % data_dir, header=True)

store_csv = spark.read.csv('%s/store.csv' % data_dir, header=True)
store_states_csv = spark.read.csv('%s/store_states.csv' % data_dir, header=True)
state_names_csv = spark.read.csv('%s/state_names.csv' % data_dir, header=True)
google_trend_csv = spark.read.csv('%s/googletrend.csv' % data_dir, header=True)
weather_csv = spark.read.csv('%s/weather.csv' % data_dir, header=True)


def expand_date(df):
    df = df.withColumn('Date', df.Date.cast(T.DateType()))
    return df \
        .withColumn('Year', F.year(df.Date)) \
        .withColumn('Month', F.month(df.Date)) \
        .withColumn('Week', F.weekofyear(df.Date)) \
        .withColumn('Day', F.dayofmonth(df.Date))


def prepare_google_trend():
    # Extract week start date and state.
    google_trend_all = google_trend_csv \
        .withColumn('Date', F.regexp_extract(google_trend_csv.week, '(.*?) -', 1)) \
        .withColumn('State', F.regexp_extract(google_trend_csv.file, 'Rossmann_DE_(.*)', 1))

    # Map state NI -> HB,NI to align with other data sources.
    google_trend_all = google_trend_all \
        .withColumn('State', F.when(google_trend_all.State == 'NI', 'HB,NI').otherwise(google_trend_all.State))

    # Expand dates.
    return expand_date(google_trend_all)


def add_elapsed(df, cols):
    def add_elapsed_column(col, asc):
        def fn(rows):
            last_store, last_date = None, None
            for r in rows:
                if last_store != r.Store:
                    last_store = r.Store
                    last_date = r.Date
                if r[col]:
                    last_date = r.Date
                fields = r.asDict().copy()
                fields[('After' if asc else 'Before') + col] = (r.Date - last_date).days
                yield Row(**fields)
        return fn

    df = df.repartition(df.Store)
    for asc in [False, True]:
        sort_col = df.Date.asc() if asc else df.Date.desc()
        rdd = df.sortWithinPartitions(df.Store.asc(), sort_col).rdd
        for col in cols:
            rdd = rdd.mapPartitions(add_elapsed_column(col, asc))
        df = rdd.toDF()
    return df


def prepare_df(df):
    num_rows = df.count()

    # Expand dates.
    df = expand_date(df)

    df = df \
        .withColumn('Open', df.Open != '0') \
        .withColumn('Promo', df.Promo != '0') \
        .withColumn('StateHoliday', df.StateHoliday != '0') \
        .withColumn('SchoolHoliday', df.SchoolHoliday != '0')

    # Merge in store information.
    store = store_csv.join(store_states_csv, 'Store')
    df = df.join(store, 'Store')

    # Merge in Google Trend information.
    google_trend_all = prepare_google_trend()
    df = df.join(google_trend_all, ['State', 'Year', 'Week']).select(df['*'], google_trend_all.trend)

    # Merge in Google Trend for whole Germany.
    google_trend_de = google_trend_all[google_trend_all.file == 'Rossmann_DE'].withColumnRenamed('trend', 'trend_de')
    df = df.join(google_trend_de, ['Year', 'Week']).select(df['*'], google_trend_de.trend_de)

    # Merge in weather.
    weather = weather_csv.join(state_names_csv, weather_csv.file == state_names_csv.StateName)
    df = df.join(weather, ['State', 'Date'])

    # Fix null values.
    df = df \
        .withColumn('CompetitionOpenSinceYear', F.coalesce(df.CompetitionOpenSinceYear, F.lit(1900))) \
        .withColumn('CompetitionOpenSinceMonth', F.coalesce(df.CompetitionOpenSinceMonth, F.lit(1))) \
        .withColumn('Promo2SinceYear', F.coalesce(df.Promo2SinceYear, F.lit(1900))) \
        .withColumn('Promo2SinceWeek', F.coalesce(df.Promo2SinceWeek, F.lit(1)))

    # Days & months competition was open, cap to 2 years.
    df = df.withColumn('CompetitionOpenSince',
                       F.to_date(F.format_string('%s-%s-15', df.CompetitionOpenSinceYear,
                                                 df.CompetitionOpenSinceMonth)))
    df = df.withColumn('CompetitionDaysOpen',
                       F.when(df.CompetitionOpenSinceYear > 1900,
                              F.greatest(F.lit(0), F.least(F.lit(360 * 2), F.datediff(df.Date, df.CompetitionOpenSince))))
                       .otherwise(0))
    df = df.withColumn('CompetitionMonthsOpen', (df.CompetitionDaysOpen / 30).cast(T.IntegerType()))

    # Days & weeks of promotion, cap to 25 weeks.
    df = df.withColumn('Promo2Since',
                       F.expr('date_add(format_string("%s-01-01", Promo2SinceYear), (cast(Promo2SinceWeek as int) - 1) * 7)'))
    df = df.withColumn('Promo2Days',
                       F.when(df.Promo2SinceYear > 1900,
                              F.greatest(F.lit(0), F.least(F.lit(25 * 7), F.datediff(df.Date, df.Promo2Since))))
                       .otherwise(0))
    df = df.withColumn('Promo2Weeks', (df.Promo2Days / 7).cast(T.IntegerType()))

    # Check that we did not lose any rows through inner joins.
    assert num_rows == df.count(), 'lost rows in joins'
    return df


def build_vocabulary(df, cols):
    vocab = {}
    for col in cols:
        values = [r[0] for r in df.select(col).distinct().collect()]
        col_type = type([x for x in values if x is not None][0])
        default_value = col_type()
        vocab[col] = sorted(values, key=lambda x: x or default_value)
    return vocab


def cast_columns(df, cols):
    for col in cols:
        df = df.withColumn(col, F.coalesce(df[col].cast(T.FloatType()), F.lit(0.0)))
    return df


def lookup_columns(df, vocab):
    def lookup(mapping):
        def fn(v):
            return mapping.index(v)
        return F.udf(fn, returnType=T.IntegerType())

    for col, mapping in vocab.items():
        df = df.withColumn(col, lookup(mapping)(df[col]))
    return df


if args.sample_rate:
    train_csv = train_csv.sample(withReplacement=False, fraction=args.sample_rate)
    test_csv = test_csv.sample(withReplacement=False, fraction=args.sample_rate)

# Prepare data frames from CSV files.
train_df = prepare_df(train_csv).cache()
test_df = prepare_df(test_csv).cache()

# Add elapsed times from holidays & promos, the data spanning training & test datasets.
elapsed_cols = ['Promo', 'StateHoliday', 'SchoolHoliday']
elapsed = add_elapsed(train_df.select('Date', 'Store', *elapsed_cols)
                              .unionAll(test_df.select('Date', 'Store', *elapsed_cols)),
                      elapsed_cols)

# Join with elapsed times.
train_df = train_df \
    .join(elapsed, ['Date', 'Store']) \
    .select(train_df['*'], *[prefix + col for prefix in ['Before', 'After'] for col in elapsed_cols])
test_df = test_df \
    .join(elapsed, ['Date', 'Store']) \
    .select(test_df['*'], *[prefix + col for prefix in ['Before', 'After'] for col in elapsed_cols])

# Filter out zero sales.
train_df = train_df.filter(train_df.Sales > 0)

print('===================')
print('Prepared data frame')
print('===================')
train_df.show()

categorical_cols = [
    'Store', 'State', 'DayOfWeek', 'Year', 'Month', 'Day', 'Week', 'CompetitionMonthsOpen', 'Promo2Weeks', 'StoreType',
    'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear', 'Events', 'Promo',
    'StateHoliday', 'SchoolHoliday'
]

continuous_cols = [
    'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity',
    'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_de',
    'BeforePromo', 'AfterPromo', 'AfterStateHoliday', 'BeforeStateHoliday', 'BeforeSchoolHoliday', 'AfterSchoolHoliday'
]

all_cols = categorical_cols + continuous_cols

# Select features.
train_df = train_df.select(*(all_cols + ['Sales', 'Date'])).cache()
test_df = test_df.select(*(all_cols + ['Id', 'Date'])).cache()

# Build vocabulary of categorical columns.
vocab = build_vocabulary(train_df.select(*categorical_cols)
                                 .unionAll(test_df.select(*categorical_cols)).cache(),
                         categorical_cols)

# Cast continuous columns to float & lookup categorical columns.
train_df = cast_columns(train_df, continuous_cols + ['Sales'])
train_df = lookup_columns(train_df, vocab)
test_df = cast_columns(test_df, continuous_cols)
test_df = lookup_columns(test_df, vocab)

# Split into training & validation.
# Test set is in 2015, use the same period in 2014 from the training set as a validation set.
test_min_date = test_df.agg(F.min(test_df.Date)).collect()[0][0]
test_max_date = test_df.agg(F.max(test_df.Date)).collect()[0][0]
a_year = datetime.timedelta(365)
val_df = train_df.filter((test_min_date - a_year <= train_df.Date) & (train_df.Date < test_max_date - a_year))
train_df = train_df.filter((train_df.Date < test_min_date - a_year) | (train_df.Date >= test_max_date - a_year))

# Determine max Sales number.
max_sales = train_df.agg(F.max(train_df.Sales)).collect()[0][0]

print('===================================')
print('Data frame with transformed columns')
print('===================================')
train_df.show()

print('================')
print('Data frame sizes')
print('================')
train_rows, val_rows, test_rows = train_df.count(), val_df.count(), test_df.count()
print('Training: %d' % train_rows)
print('Validation: %d' % val_rows)
print('Test: %d' % test_rows)

# Save data frames as Parquet files.
train_df.write.parquet('%s/train_df.parquet' % data_dir, mode='overwrite')
val_df.write.parquet('%s/val_df.parquet' % data_dir, mode='overwrite')
test_df.write.parquet('%s/test_df.parquet' % data_dir, mode='overwrite')

# Finish the data preparation
spark.stop()


print('==============')
print('Model training')
print('==============')

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Flatten, Reshape, BatchNormalization, Dropout
import tensorflow.keras.backend as K


def exp_rmspe(y_true, y_pred):
    """Competition evaluation metric, expects logarithic inputs."""
    pct = tf.square((tf.exp(y_true) - tf.exp(y_pred)) / tf.exp(y_true))
    # Compute mean excluding stores with zero denominator.
    x = tf.reduce_sum(tf.where(y_true > 0.001, pct, tf.zeros_like(pct)))
    y = tf.reduce_sum(tf.where(y_true > 0.001, tf.ones_like(pct), tf.zeros_like(pct)))
    return tf.sqrt(x / y)


def act_sigmoid_scaled(x):
    """Sigmoid scaled to logarithm of maximum sales scaled by 20%."""
    return tf.nn.sigmoid(x) * tf.math.log(max_sales) * 1.2


CUSTOM_OBJECTS = {'exp_rmspe': exp_rmspe,
                  'act_sigmoid_scaled': act_sigmoid_scaled}


def serialize_model(model):
    """Serialize model into byte array."""
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        model.save(f)
    return bio.getvalue()


def deserialize_model(model_bytes, load_model_fn):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio, 'r') as f:
        return load_model_fn(f, custom_objects=CUSTOM_OBJECTS)


# Disable GPUs when building the model to prevent memory leaks
if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
    # See https://github.com/tensorflow/tensorflow/issues/33168
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    K.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))


# Build the model.
def build_model():
    inputs = {col: Input(shape=(1,), name=col) for col in all_cols}
    embeddings = [Embedding(len(vocab[col]), 10, input_length=1, name='emb_' + col)(inputs[col])
                  for col in categorical_cols]
    continuous_bn = Concatenate()([Reshape((1, 1), name='reshape_' + col)(inputs[col])
                                   for col in continuous_cols])
    continuous_bn = BatchNormalization()(continuous_bn)
    x = Concatenate()(embeddings + [continuous_bn])
    x = Flatten()(x)
    x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation=act_sigmoid_scaled)(x)
    model = tf.keras.Model([inputs[f] for f in all_cols], output)
    model.summary()
    return model


# Horovod: compile keras model with Horovod
import horovod.spark
import horovod.tensorflow.keras as hvd


def compile_horovod_model(learning_rate):
    model = build_model()

    # Horovod: add Distributed Optimizer.
    opt = tf.keras.optimizers.Adam(lr=learning_rate, epsilon=1e-3)
    opt = hvd.DistributedOptimizer(opt)
    model.compile(opt, 'mae', metrics=[exp_rmspe])
    model_bytes = serialize_model(model)
    return model_bytes


#  Horovod: distributed training
# Set the parameters
num_proc = args.num_proc if args.num_proc else total_workers
print("Train processes: {}".format(num_proc))

batch_size = args.batch_size
print("Train batch size: {}".format(batch_size))

epochs = args.epochs
print("Train epochs: {}".format(epochs))


def train_horovod(model_bytes):
    # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
    # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
    # functions like `deserialize_model` which are implemented at the top level.
    # See https://jira.apache.org/jira/browse/ARROW-3346
    pa

    import atexit
    import horovod.tensorflow.keras as hvd
    import os
    from petastorm import make_batch_reader
    from petastorm.tf_utils import make_petastorm_dataset
    import tempfile
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import shutil

    # Horovod: initialize Horovod inside the trainer.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process), if GPUs are available.
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    # K.set_session(tf.compat.v1.Session(config=config))

    # Horovod: restore from checkpoint, use hvd.load_model under the hood.
    model = deserialize_model(model_bytes, hvd.load_model)

    # Horovod: adjust learning rate based on number of processes.
    scaled_lr = K.get_value(model.optimizer.lr) * hvd.size()
    K.set_value(model.optimizer.lr, scaled_lr)

    # Horovod: print summary logs on the first worker.
    verbose = 2 if hvd.rank() == 0 else 0

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=5, verbose=verbose),

        # Reduce LR if the metric is not improved for 10 epochs, and stop training
        # if it has not improved for 20 epochs.
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_exp_rmspe', patience=10, verbose=verbose),
        tf.keras.callbacks.EarlyStopping(monitor='val_exp_rmspe', mode='min', patience=20, verbose=verbose),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # Model checkpoint location.
    ckpt_dir = tempfile.mkdtemp()
    ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='val_exp_rmspe', mode='min',
                                                            save_best_only=True))

    # Make Petastorm readers.
    with make_batch_reader('%s/train_df.parquet' % data_dir, num_epochs=None,
                           cur_shard=hvd.rank(), shard_count=hvd.size(),
                           hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
        with make_batch_reader('%s/val_df.parquet' % data_dir, num_epochs=None,
                               cur_shard=hvd.rank(), shard_count=hvd.size(),
                               hdfs_driver=PETASTORM_HDFS_DRIVER) as val_reader:
            # Convert readers to tf.data.Dataset.
            train_ds = make_petastorm_dataset(train_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .shuffle(int(train_rows / hvd.size())) \
                .batch(batch_size) \
                .map(lambda x: (tuple(getattr(x, col) for col in all_cols), tf.math.log(x.Sales)))

            val_ds = make_petastorm_dataset(val_reader) \
                .apply(tf.data.experimental.unbatch()) \
                .batch(batch_size) \
                .map(lambda x: (tuple(getattr(x, col) for col in all_cols), tf.math.log(x.Sales)))

            history = model.fit(train_ds,
                                validation_data=val_ds,
                                steps_per_epoch=int(train_rows / batch_size / hvd.size()),
                                validation_steps=int(val_rows / batch_size / hvd.size()),
                                callbacks=callbacks,
                                verbose=verbose,
                                epochs=epochs)

    # Dataset API usage currently displays a wall of errors upon termination.
    # This global model registration ensures clean termination.
    # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
    globals()['_DATASET_FINALIZATION_HACK'] = model

    if hvd.rank() == 0:
        with open(ckpt_file, 'rb') as f:
            return history.history, f.read()


# Create Spark session for training.
conf = SparkConf().setAppName('spark-keras-rossmann-training')
if args.training_master:
    conf.setMaster(args.training_master)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

checkpoint_dir = create_log_dir('rossmann-keras')
print("Log directory:", checkpoint_dir)


# Horovod: run training.
def train(learning_rate):
    model_bytes = compile_horovod_model(learning_rate)
    history, trained_model_bytes = \
        horovod.spark.run(train_horovod, args=(model_bytes,), num_proc=num_proc,
                          stdout=sys.stdout, stderr=sys.stderr, verbose=2,
                          prefix_output_with_timestamp=True)[0]

    best_val_rmspe = min(history['val_exp_rmspe'])
    print('Best RMSPE: %f' % best_val_rmspe)

    # Write checkpoint.
    local_checkpoint_file = get_checkpoint_file(checkpoint_dir, learning_rate)
    with open(local_checkpoint_file, 'wb') as f:
        f.write(trained_model_bytes)
    print('Written checkpoint to %s' % local_checkpoint_file)
    return trained_model_bytes, best_val_rmspe


def load_model_of_checkpoint(log_dir, file_id):
    checkpoint_file = get_checkpoint_file(log_dir, file_id)
    # Restore from checkpoint.
    with open(checkpoint_file, 'rb') as f:
        model_bytes = f.read()
        return deserialize_model(model_bytes, tf.keras.models.load_model), model_bytes


#  Hyperopt training function
import mlflow


def hyper_objective(learning_rate):
    with mlflow.start_run():
        model_bytes, test_loss = train(learning_rate)

        mlflow.log_metric("learning_rate", learning_rate)
        mlflow.log_metric("loss", test_loss)
    return {'loss': test_loss, 'status': STATUS_OK}


# Do a super parameter tuning with hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

trials = args.trials
print("Hyper parameter tuning trials: {}".format(trials))

search_space = hp.uniform('learning_rate', 0.0001, 0.001)
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("Rossmann: Keras + Spark + Horovod")
argmin = fmin(
    fn=hyper_objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=trials)
print("Best parameter found: ", argmin)


# Load final model with the best parameters
best_model, best_model_bytes = load_model_of_checkpoint(checkpoint_dir, argmin.get('learning_rate'))
model_name = 'rossmann-keras-run-model'
mlflow.keras.log_model(best_model, model_name, registered_model_name=model_name)


# Load the model from MLflow and run a transformation
model_uri = "models:/{}/latest".format(model_name)
print('Inference with model: {}'.format(model_uri))
saved_model = mlflow.keras.load_model(model_uri, custom_objects=CUSTOM_OBJECTS)


# Stop the training Spark session
spark.stop()


# Do predictions
# Create Spark session for prediction.
conf = SparkConf().setAppName('spark-keras-rossmann-prediction') \
    .setExecutorEnv('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH')) \
    .setExecutorEnv('PATH', os.environ.get('PATH'))
spark = SparkSession.builder.config(conf=conf).getOrCreate()


def predict_fn(model_bytes):
    def fn(rows):
        import math
        import tensorflow as tf
        import tensorflow.compat.v1.keras.backend as K
        import numpy as np

        # Do not use GPUs for prediction, use single CPU core per task.
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        config.inter_op_parallelism_threads = 1
        config.intra_op_parallelism_threads = 1
        K.set_session(tf.compat.v1.Session(config=config))

        # Restore from checkpoint.
        model = deserialize_model(model_bytes, tf.keras.models.load_model)

        # Perform predictions.
        for row in rows:
            fields = row.asDict().copy()
            # Convert from log domain to real Sales numbers.
            log_sales = model.predict_on_batch([
                np.array([row[col]]) for col in all_cols
            ])[0]
            # Set 'Sales_pred' column with prediction results.
            fields['Sales_output'] = log_sales[0].item()
            fields['Sales_pred'] = math.exp(log_sales)
            yield Row(**fields)

    return fn


def df_rmspe(pred_df):
    rmspe_df = pred_df.withColumn('pct', F.pow((pred_df.Sales - pred_df.Sales_pred) / pred_df.Sales, 2))
    rmspe_df = rmspe_df.filter(rmspe_df.Sales_output > 0.001)
    sum_pct = rmspe_df.agg(F.sum(rmspe_df.pct)).collect()[0][0]
    count_pct = rmspe_df.count()
    rmspe = math.sqrt(sum_pct / count_pct)
    return rmspe


def test_model(model, show_samples=False):
    model_bytes = serialize_model(model)
    pred_df = spark.read.parquet('%s/val_df.parquet' % data_dir) \
        .rdd.mapPartitions(predict_fn(model_bytes)).toDF()

    # Compute the RMSPE
    rmspe = df_rmspe(pred_df)
    print('Test RMSPE: %f' % rmspe)

    if show_samples:
        pred_df.show(5)
    return rmspe


# Test model
test_model(saved_model, True)


print('============================')
print('Final prediction submission')
print('============================')

# Submit a Spark job to do inference. Horovod framework is not involved here.
pred_df = spark.read.parquet('%s/test_df.parquet' % data_dir) \
    .rdd.mapPartitions(predict_fn(best_model_bytes)).toDF()
pred_df = pred_df.withColumn('Sales', pred_df.Sales_pred)
submission_df = pred_df.select(pred_df.Id.cast(T.IntegerType()), pred_df.Sales).toPandas()
submission_df.sort_values(by=['Id']).to_csv(args.local_submission_csv, index=False)
print('Saved predictions to %s' % args.local_submission_csv)


# Stop the prediction Spark session
spark.stop()
