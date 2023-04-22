"""The tags/labels to associate metadata with instances."""

# Tag for the name of the node
CLOUDTIK_TAG_NODE_NAME = "cloudtik-node-name"

# Tag uniquely identifying all nodes of a cluster
CLOUDTIK_TAG_CLUSTER_NAME = "cloudtik-cluster-name"

# Tag for the kind of node (e.g. Head, Worker).
CLOUDTIK_TAG_NODE_KIND = "cloudtik-node-kind"
NODE_KIND_HEAD = "head"
NODE_KIND_WORKER = "worker"
NODE_KIND_UNMANAGED = "unmanaged"

# Tag for user defined node types (e.g., m4xl_spot). This is used for multi
# node type clusters.
CLOUDTIK_TAG_USER_NODE_TYPE = "cloudtik-user-node-type"

# Tag that reports the current state of the node (e.g. Updating, Up-to-date)
CLOUDTIK_TAG_NODE_STATUS = "cloudtik-node-status"
STATUS_UNINITIALIZED = "uninitialized"
STATUS_WAITING_FOR_SSH = "waiting-for-ssh"
STATUS_BOOTSTRAPPING_DATA_DISKS = "bootstrapping-data-disks"
STATUS_SYNCING_FILES = "syncing-files"
STATUS_SETTING_UP = "setting-up"
STATUS_UPDATE_FAILED = "update-failed"
STATUS_UP_TO_DATE = "up-to-date"

# Hash of the node launch config, used to identify out-of-date nodes
CLOUDTIK_TAG_LAUNCH_CONFIG = "cloudtik-launch-config"

# Hash of the node runtime config, used to determine if updates are needed
CLOUDTIK_TAG_RUNTIME_CONFIG = "cloudtik-runtime-config"

# Hash of the contents of the directories specified by the file_mounts config
# if the node is a worker, this also hashes content of the directories
# specified by the cluster_synced_files config
CLOUDTIK_TAG_FILE_MOUNTS_CONTENTS = "cloudtik-file-mounts-contents"

# The prefix used for global variables published to workspace
CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX = "cloudtik-global-"
CLOUDTIK_GLOBAL_VARIABLE_KEY = CLOUDTIK_GLOBAL_VARIABLE_KEY_PREFIX + "{}"

# The cluster wide unique numeric id for the node
CLOUDTIK_TAG_NODE_NUMBER = "cloudtik-node-number"
# The head node will always be assigned with number 1
CLOUDTIK_TAG_HEAD_NODE_NUMBER = 1

# Quorum generation of the node when managing a Quorum of minimal nodes
CLOUDTIK_TAG_QUORUM_ID = "cloudtik-quorum-id"
