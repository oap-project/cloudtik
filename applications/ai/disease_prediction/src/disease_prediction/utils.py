import yaml


DEFAULT_TRAIN_OUTPUT = "train_output.yaml"
DEFAULT_PREDICT_OUTPUT = "predict_output.yaml"


def read_yaml_file(yaml_path):
    with open(yaml_path, "r") as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict


def parse_arguments(parser, arguments):
    if isinstance(arguments, str):
        if arguments.endswith(".json"):
            args = parser.parse_json_file(json_file=arguments)
        else:
            # default a yaml file
            with open(arguments, "r") as f:
                args_in_yaml = yaml.safe_load(f)
            args = parser.parse_dict(args=args_in_yaml)
    elif isinstance(arguments, dict):
        # a dict
        args = parser.parse_dict(args=arguments)
    else:
        # it is already an arguments object
        return arguments

    return args


def get_subject_id(image_name):
    """
    Extracts the patient ID from an image filename.

    Args:
    - image_name: string representing the filename of an image

    Returns:
    - patient_id: string representing the patient ID extracted from the image filename
    """

    # Split the filename by "/"
    image_name = image_name.split("/")[-1]

    # Extract the first two substrings separated by "_", remove the first character (which is "P"), and join them
    # together to form the patient ID
    patient_id = "".join(image_name.split("_")[:2])[1:]

    return patient_id
