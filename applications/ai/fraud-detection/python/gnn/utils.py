import argparse
import os


def existing_directory(raw_path):
    if not os.path.isdir(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing directory'.format(raw_path)
        )
    return os.path.abspath(raw_path)


def existing_file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)
