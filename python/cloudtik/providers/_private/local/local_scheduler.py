import logging

logger = logging.getLogger(__name__)


class LocalScheduler:
    def __init__(self, provider_config):
        self.provider_config = provider_config

    def create_node(self, cluster_name, node_config, tags, count):
        raise NotImplementedError

    def get_non_terminated_nodes(self, tag_filters):
        raise NotImplementedError

    def is_running(self, node_id):
        raise NotImplementedError

    def is_terminated(self, node_id):
        raise NotImplementedError

    def get_node_tags(self, node_id):
        raise NotImplementedError

    def get_internal_ip(self, node_id):
        raise NotImplementedError

    def set_node_tags(self, node_id, tags):
        raise NotImplementedError

    def terminate_node(self, node_id):
        raise NotImplementedError

    def get_node_info(self, node_id):
        raise NotImplementedError
