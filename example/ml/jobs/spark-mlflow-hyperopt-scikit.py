from cloudtik.core.api import ThisCluster

cluster = ThisCluster()
cluster_head_ip = cluster.get_head_node_ip()
# Wait for all cluster works read
cluster.wait_for_ready()


# Initialize SparkSession
from pyspark import SparkConf
from pyspark.sql import SparkSession

spark_conf = SparkConf().setAppName('spark-scikit').set('spark.sql.shuffle.partitions', '16')
spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
conf = spark.conf


# Load the iris dataset from scikit-learn
from sklearn.datasets import load_iris

iris = iris = load_iris()
X = iris.data
y = iris.target


# Define a train function and objective to minimize.
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# Function to train a model
def train(C):
    # Create a support vector classifier model
    model = SVC(C=C)
    model.fit(X, y)
    return model


# Objective function to minimize
def hyper_objective(C):
    # Create a support vector classifier model
    model = train(C)

    # Use the cross-validation accuracy to compare the models' performance
    accuracy = cross_val_score(model, X, y).mean()
    with mlflow.start_run():
        mlflow.log_metric("C", C)
        mlflow.log_metric("loss", -accuracy)

    # Hyperopt tries to minimize the objective function.
    # A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': -accuracy, 'C': C, 'status': STATUS_OK}


# Do a super parameter tuning with hyperopt
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import mlflow

# Define the search space and select a search algorithm
search_space = hp.lognormal('C', 0, 1.0)
algo = tpe.suggest
spark_trials = SparkTrials(spark_session=spark)

mlflow.set_tracking_uri(f"http://{cluster_head_ip}:5001")
mlflow.set_experiment("MLflow + HyperOpt + Scikit-Learn")
argmin = fmin(
  fn=hyper_objective,
  space=search_space,
  algo=algo,
  max_evals=16,
  trials=spark_trials)

# Print the best value found for C
print("Best value found: ", argmin)
print("argmin.get('C'): ", argmin.get('C'))


# Train final model with the best parameters
best_model = train(argmin.get('C'))
mlflow.sklearn.log_model(best_model, "Sklearn-SVC-model",
                         registered_model_name="Sklearn-SVC-model-reg")
model_uri = "models:/Sklearn-SVC-model-reg/1"


# Load model as a PyFuncModel and predict on a Pandas DataFrame.
import pandas as pd

loaded_model = mlflow.pyfunc.load_model(model_uri)
loaded_model.predict(pd.DataFrame(X))


# Clean up
spark.stop()
