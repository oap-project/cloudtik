

def to_label_selector(tags):
    label_selector = ""
    for k, v in tags.items():
        if label_selector != "":
            label_selector += ","
        label_selector += "{}={}".format(k, v)
    return label_selector


def _get_node_info(pod):
    node_info = {"node_id": pod.metadata.name,
                 "instance_type": "Unknown",
                 "private_ip": pod.status.pod_ip,
                 "public_ip": None,
                 "instance_status": pod.status.phase}
    node_info.update(pod.metadata.labels)

    return node_info
