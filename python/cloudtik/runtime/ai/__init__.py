import os

# define the runtime script alias which can be run simply
# avoid to use import so that the dependencies will not cause error
_script_aliases_ = {
    "ai.launch": os.path.join(
        os.path.dirname(__file__),
        "runner/launch.py"),
    "ai.modeling.graph-sage": os.path.join(
        os.path.dirname(__file__),
        "modeling/graph_modeling/graph_sage/modeling/run.py"),
    "ai.modeling.xgboost": os.path.join(
        os.path.dirname(__file__),
        "modeling/classical_ml/classification_and_regression/xgboost/modeling/run.py"),
}
