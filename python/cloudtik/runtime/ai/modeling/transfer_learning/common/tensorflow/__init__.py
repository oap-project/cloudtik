import os


def get_train_script():
    this_dir = os.path.dirname(__file__)
    train_script = os.path.join(this_dir, "train.py")
    return train_script
