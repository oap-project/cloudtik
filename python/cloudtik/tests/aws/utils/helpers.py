import os
import yaml
import cloudtik
import copy

from cloudtik.providers._private.aws.node_provider import AWSNodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_NODE_KIND, NODE_KIND_HEAD, \
    NODE_KIND_WORKER, CLOUDTIK_TAG_USER_NODE_TYPE, CLOUDTIK_TAG_CLUSTER_NAME
from cloudtik.core._private.utils import prepare_config, validate_config
from cloudtik.tests.aws.utils.constants import DEFAULT_CLUSTER_NAME, \
    DEFAULT_NODE_PROVIDER_INSTANCE_TAGS


def get_aws_example_config_file_path(file_name):
    import cloudtik.providers.aws
    return os.path.join(
        os.path.dirname(cloudtik.providers.aws.__file__), file_name)


def load_aws_example_config_file(file_name):
    config_file_path = get_aws_example_config_file_path(file_name)
    return yaml.safe_load(open(config_file_path).read())


def bootstrap_aws_config(config):
    config = prepare_config(config)
    validate_config(config)
    config["cluster_name"] = DEFAULT_CLUSTER_NAME
    return cloudtik.providers._private.aws.config.bootstrap_aws(config)


def bootstrap_aws_example_config_file(file_name):
    config = load_aws_example_config_file(file_name)
    return bootstrap_aws_config(config)


def node_provider_tags(config, type_name):
    """
    Returns a copy of DEFAULT_NODE_PROVIDER_INSTANCE_TAGS with the CloudTik node
    kind and CloudTik user node type filled in from the input config and node type
    name.

    Args:
        config: cluster config
        type_name: node type name
    Returns:
        tags: node provider tags
    """
    tags = copy.copy(DEFAULT_NODE_PROVIDER_INSTANCE_TAGS)
    head_name = config["head_node_type"]
    node_kind = NODE_KIND_HEAD if type_name is head_name else NODE_KIND_WORKER
    tags[CLOUDTIK_TAG_NODE_KIND] = node_kind
    tags[CLOUDTIK_TAG_USER_NODE_TYPE] = type_name
    return tags


def apply_node_provider_config_updates(config, node_cfg, node_type_name,
                                       max_count):
    """
    Applies default updates made by AWSNodeProvider to node_cfg during node
    creation. This should only be used for testing purposes.

    Args:
        config: cluster config
        node_cfg: node config
        node_type_name: node type name
        max_count: max nodes of the given type to launch
    """
    tags = node_provider_tags(config, node_type_name)
    tags[CLOUDTIK_TAG_CLUSTER_NAME] = DEFAULT_CLUSTER_NAME
    user_tag_specs = node_cfg.get("TagSpecifications", [])
    tag_specs = [{
        "ResourceType": "instance",
        "Tags": [{
            "Key": k,
            "Value": v
        } for k, v in sorted(tags.items())]
    }]
    node_provider_cfg_updates = {
        "MinCount": 1,
        "MaxCount": max_count,
        "TagSpecifications": tag_specs,
    }
    tags.pop(CLOUDTIK_TAG_CLUSTER_NAME)
    node_cfg.update(node_provider_cfg_updates)
    # merge node provider tag specs with user overrides
    AWSNodeProvider._merge_tag_specs(tag_specs, user_tag_specs)
