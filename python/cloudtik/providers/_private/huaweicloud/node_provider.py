import copy
import logging
from threading import RLock
from typing import Any, Dict, List, Optional

from huaweicloudsdkecs.v2 import BatchCreateServerTagsRequest, \
    BatchCreateServerTagsRequestBody, CreatePostPaidServersRequest, \
    CreatePostPaidServersRequestBody, DeleteServersRequest, \
    DeleteServersRequestBody, \
    ListServersDetailsRequest, \
    PostPaidServer, PostPaidServerDataVolume, PostPaidServerEip, \
    PostPaidServerEipBandwidth, PostPaidServerExtendParam, \
    PostPaidServerNic, \
    PostPaidServerPublicip, PostPaidServerRootVolume, \
    PostPaidServerSecurityGroup, PostPaidServerTag, ServerId, ServerTag

from cloudtik.core._private.cli_logger import cli_logger
from cloudtik.core._private.utils import \
    get_cluster_node_public_ip_bandwidth_conf
from cloudtik.core.node_provider import NodeProvider
from cloudtik.core.tags import CLOUDTIK_TAG_CLUSTER_NAME, \
    CLOUDTIK_TAG_NODE_NAME
from cloudtik.providers._private.huaweicloud.config import \
    bootstrap_huaweicloud, post_prepare_huaweicloud, verify_obs_storage, \
    with_huaweicloud_environment_variables
from cloudtik.providers._private.huaweicloud.utils import _get_node_info, \
    _get_node_private_and_public_ip, _make_ecs_client, \
    flat_tags_map, get_default_huaweicloud_cloud_storage, \
    get_huaweicloud_obs_storage_config, \
    HWC_SERVER_STATUS_ACTIVE, \
    HWC_SERVER_STATUS_NON_TERMINATED, HWC_SERVER_TAG_STR_FORMAT, \
    tags_list_to_dict
from cloudtik.providers._private.utils import validate_config_dict

logger = logging.getLogger(__name__)


def synchronized(f):
    def wrapper(self, *args, **kwargs):
        self.lock.acquire()
        try:
            return f(self, *args, **kwargs)
        finally:
            self.lock.release()

    return wrapper


class HUAWEICLOUDNodeProvider(NodeProvider):
    def __init__(self, provider_config, cluster_name):
        NodeProvider.__init__(self, provider_config, cluster_name)
        self.ecs_client = _make_ecs_client(self.provider_config)
        self.lock = RLock()
        # Cache of node objects from the last API call. This avoids
        # excessive listServerDetails() requests.
        self.cached_nodes = {}

    def with_environment_variables(self, node_type_config: Dict[str, Any],
                                   node_id: str):
        """Export necessary environment variables for running node commands"""
        return with_huaweicloud_environment_variables(self.provider_config,
                                                      node_type_config,
                                                      node_id)

    def get_default_cloud_storage(self):
        """Return the managed cloud storage if configured."""
        return get_default_huaweicloud_cloud_storage(self.provider_config)

    @staticmethod
    def post_prepare(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fills out missing fields after the user config is merged
        with defaults. This happens after prepare_config is done.
        """
        return post_prepare_huaweicloud(cluster_config)

    @staticmethod
    def validate_config(provider_config: Dict[str, Any]) -> None:
        """Check the provider configuration validation.
        This happens after post_prepare is done and before bootstrap_config
        """
        config_dict = {"region": provider_config.get("region")}
        validate_config_dict(provider_config["type"], config_dict)

        storage_config = get_huaweicloud_obs_storage_config(provider_config)
        if storage_config is not None:
            config_dict = {
                "obs.bucket": storage_config.get("obs.bucket"),
                # FIXME(ChenRui): Huawei cloud obsfs don't support to mount
                # fuse with server agency, have to config AK/SK explicitly at
                # CloudTik side, see details as follows:
                # https://github.com/huaweicloud/huaweicloud-obs-obsfs/issues/8
                # "obs.access.key": storage_config.get("obs.access.key"),
                # "obs.secret.key": storage_config.get("obs.secret.key")
            }

            validate_config_dict(provider_config["type"], config_dict)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Bootstraps the cluster config by adding env defaults if needed.
        This happens after validate_config is done.
        """
        return bootstrap_huaweicloud(cluster_config)

    @staticmethod
    def verify_config(provider_config: Dict[str, Any]) -> None:
        """Verify provider configuration. Verification usually means to check
        it is working. This happens after bootstrap_config is done.
        """
        verify_cloud_storage = provider_config.get("verify_cloud_storage",
                                                   True)
        cloud_storage = get_huaweicloud_obs_storage_config(provider_config)
        if verify_cloud_storage and cloud_storage is not None:
            cli_logger.verbose("Verifying OBS storage configurations...")
            verify_obs_storage(provider_config)
            cli_logger.verbose(
                "Successfully verified OBS storage configurations.")

    @synchronized
    def _get_filtered_nodes(self, tag_filters):
        # cluster scope servers
        tags_str = flat_tags_map(
            self._copy_tags_with_cluster_name(tag_filters))
        # Huawei cloud don't return terminated instances
        nodes = self.ecs_client.list_servers_details(
            ListServersDetailsRequest(tags=tags_str)).servers
        # update cached
        self.cached_nodes = {node.id: node for node in nodes}
        return self.cached_nodes

    def non_terminated_nodes(self, tag_filters: Dict[str, str]) -> List[str]:
        nodes = self._get_filtered_nodes(tag_filters)
        return [node_id for node_id, node in nodes.items() if
                node.status in HWC_SERVER_STATUS_NON_TERMINATED]

    def _copy_tags_with_cluster_name(self, tags):
        _tags = {CLOUDTIK_TAG_CLUSTER_NAME: self.cluster_name}
        _tags.update(tags)
        return _tags

    def create_node(self, node_config: Dict[str, Any], tags: Dict[str, str],
                    count: int) -> Optional[Dict[str, Any]]:
        """Creates servers.

        Returns dict mapping server id to ecs.Server object for the created
        servers.
        """
        # TODO(ChenRui): reuse stopping servers
        _node_config = copy.deepcopy(node_config)
        _tags = self._copy_tags_with_cluster_name(tags)
        _node_config['name'] = _tags.get(CLOUDTIK_TAG_NODE_NAME,
                                         self.cluster_name)
        _node_config['count'] = count
        # convert str to object
        root_vol_args = _node_config.pop('root_volume')
        _node_config['root_volume'] = PostPaidServerRootVolume(
            **root_vol_args)
        data_vols_args = _node_config.pop('data_volumes', [])
        _node_config['data_volumes'] = [PostPaidServerDataVolume(**data_vol)
                                        for data_vol in data_vols_args]
        nics = _node_config.pop('nics')
        _node_config['nics'] = [PostPaidServerNic(subnet_id=nic['subnet_id'])
                                for nic in nics]
        publicip = _node_config.pop('publicip')
        if publicip:
            bandwidth = get_cluster_node_public_ip_bandwidth_conf(
                self.provider_config)
            _node_config['publicip'] = PostPaidServerPublicip(
                eip=PostPaidServerEip(
                    iptype='5_bgp',
                    bandwidth=PostPaidServerEipBandwidth(sharetype='PER',
                                                         size=bandwidth)
                ),
                delete_on_termination=True
            )
        sgs = _node_config.pop('security_groups')
        _node_config['security_groups'] = [
            PostPaidServerSecurityGroup(id=sg['id']) for sg in sgs]
        # spot nodes
        if 'extendparam' in _node_config:
            extendparam = _node_config.pop('extendparam')
            _node_config['extendparam'] = PostPaidServerExtendParam(
                **extendparam)
        if _tags:
            _node_config['server_tags'] = [PostPaidServerTag(key=k, value=v)
                                           for k, v in _tags.items()]

        server_ids = self.ecs_client.create_post_paid_servers(
            CreatePostPaidServersRequest(body=CreatePostPaidServersRequestBody(
                server=PostPaidServer(**_node_config)))).server_ids
        created_nodes_dict = {}
        for server_id in server_ids:
            created_server = self.ecs_client.list_servers_details(
                ListServersDetailsRequest(server_id=server_id)).servers[0]
            created_nodes_dict[created_server.id] = created_server
        return created_nodes_dict

    def prepare_for_head_node(self, cluster_config: Dict[str, Any],
                              remote_config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a new cluster config with custom configs for head node."""
        # Since the head will use the ECS agency to access cloud,
        # remove the client credentials from config
        if "huaweicloud_credentials" in remote_config["provider"]:
            remote_config.pop("huaweicloud_credentials", None)

        return remote_config

    def _get_node(self, node_id):
        """Refresh cache and return latest node details from cache"""
        self._get_filtered_nodes({})  # Side effect: updates cache
        return self.cached_nodes[node_id]

    def _get_cached_node(self, node_id):
        """Try to get node from cache, fallback to cloud provider if missing,
        and refresh cache."""
        if node_id in self.cached_nodes:
            return self.cached_nodes[node_id]
        return self._get_node(node_id)

    def set_node_tags(self, node_id: str, tags: Dict[str, str]) -> None:
        node = self._get_cached_node(node_id)
        need_create_tags = []
        # node.tags format: ['key1=value1', 'key2=value2', ...]
        for k, v in tags.items():
            kv_str = HWC_SERVER_TAG_STR_FORMAT.format(k, v)
            if kv_str not in node.tags:
                need_create_tags.append(ServerTag(k, v))
        if need_create_tags:
            _body = BatchCreateServerTagsRequestBody(action='create',
                                                     tags=need_create_tags)
            self.ecs_client.batch_create_server_tags(
                BatchCreateServerTagsRequest(server_id=node.id, body=_body)
            )
        # refresh node cache to update tags
        self._get_filtered_nodes({})

    def get_node_info(self, node_id: str) -> Dict[str, str]:
        node = self._get_cached_node(node_id)
        return _get_node_info(node)

    def is_running(self, node_id: str) -> bool:
        node = self._get_cached_node(node_id)
        return node.status == HWC_SERVER_STATUS_ACTIVE

    def is_terminated(self, node_id: str) -> bool:
        node = self._get_cached_node(node_id)
        return node.status not in HWC_SERVER_STATUS_NON_TERMINATED

    def node_tags(self, node_id: str) -> Dict[str, str]:
        node = self._get_cached_node(node_id)
        return tags_list_to_dict(node.tags)

    def external_ip(self, node_id: str) -> str:
        node = self._get_cached_node(node_id)
        _, public_ip = _get_node_private_and_public_ip(node)
        if not public_ip:
            node = self._get_node(node_id)
            _, public_ip = _get_node_private_and_public_ip(node)
        return public_ip

    def internal_ip(self, node_id: str) -> str:
        node = self._get_cached_node(node_id)
        private_ip, _ = _get_node_private_and_public_ip(node)
        if not private_ip:
            node = self._get_node(node_id)
            private_ip, _ = _get_node_private_and_public_ip(node)
        return private_ip

    def terminate_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        try:
            self._get_cached_node(node_id)
        except KeyError:
            # node no longer exist
            return
        # TODO(ChenRui): reuse stopping servers
        try:
            target_server = [ServerId(node_id), ]
            self.ecs_client.delete_servers(DeleteServersRequest(
                DeleteServersRequestBody(delete_publicip=True,
                                         delete_volume=True,
                                         servers=target_server)))
        except Exception as e:
            cli_logger.warning(
                "Failed to delete server {}: {}".format(node_id, e))
        # refresh node cache
        self._get_filtered_nodes({})

    @property
    def max_terminate_nodes(self) -> Optional[int]:
        return 1000

    def terminate_nodes(self, node_ids: List[str]) -> Optional[Dict[str, Any]]:
        target_servers = set()
        for node_id in node_ids:
            cli_logger.verbose(
                "NodeProvider cluster {}: Terminating node {}".format(
                    self.cluster_name, node_id))
            try:
                self._get_cached_node(node_id)
                target_servers.add(node_id)
            except KeyError:
                # node no longer exist
                pass
        batch_delete_num = min(len(target_servers), self.max_terminate_nodes)
        batch_delete = []
        try:
            for server_id in target_servers:
                batch_delete.append(ServerId(server_id))
                # handle one batch
                if len(batch_delete) % batch_delete_num == 0:
                    self.ecs_client.delete_servers(DeleteServersRequest(
                        DeleteServersRequestBody(delete_publicip=True,
                                                 delete_volume=True,
                                                 servers=batch_delete)))
                    batch_delete.clear()
            # handle reminder
            if batch_delete:
                self.ecs_client.delete_servers(DeleteServersRequest(
                    DeleteServersRequestBody(delete_publicip=True,
                                             delete_volume=True,
                                             servers=batch_delete)))
        except Exception as e:
            cli_logger.warning(
                "Failed to delete servers {}: {}".format(node_ids, e))
        # refresh node cache
        self._get_filtered_nodes({})
