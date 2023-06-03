

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
