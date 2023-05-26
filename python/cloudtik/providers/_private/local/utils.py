
def get_instance_type_name(instance_type):
    instance_type_name = instance_type.get("name")
    if instance_type_name:
        return

    # combine a name from CPU, memory
    num_cpus = instance_type.get("CPU", 0)
    memory_gb = instance_type.get("memory", 0)
    if num_cpus and memory_gb:
        return "{}CPU/{}GB".format(num_cpus, memory_gb)
    elif num_cpus:
        return "{}CPU".format(num_cpus)
    elif memory_gb:
        return "{}GB".format(memory_gb)
    return "Unknown"


def _get_tags(node):
    if node is None:
        return {}
    return node.get("tags", {})


def _get_node_info(node):
    node_info = {"node_id": node["name"],
                 "instance_type": node.get("instance_type"),
                 "private_ip": node["ip"],
                 "public_ip": None,
                 "instance_status": node["state"]}
    labels = _get_tags(node)
    node_info.update(labels)
    return node_info
