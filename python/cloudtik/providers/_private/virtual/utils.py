from cloudtik.core._private.core_utils import get_memory_in_bytes, format_memory


def get_instance_type_name(instance_type):
    instance_type_name = instance_type.get("name")
    if instance_type_name:
        return

    # combine a name from CPU, memory
    num_cpus = instance_type.get("CPU", 0)
    memory = get_memory_in_bytes(
        instance_type.get("memory", 0))
    memory_str = format_memory(memory)
    if num_cpus and memory:
        return "{}CPU/{}".format(num_cpus, memory_str)
    elif num_cpus:
        return "{}CPU".format(num_cpus)
    elif memory:
        return "{}".format(memory_str)
    return "Unknown"


def _get_tags(node):
    if node is None:
        return {}
    return node.get("tags", {})


def _get_node_info(node):
    instance_type = node.get("instance_type", {})
    node_instance_type = get_instance_type_name(instance_type)
    node_info = {"node_id": node["name"],
                 "instance_type": node_instance_type,
                 "private_ip": node["ip"],
                 "public_ip": node.get("external_ip"),
                 "instance_status": node["state"]}
    labels = _get_tags(node)
    node_info.update(labels)
    return node_info

