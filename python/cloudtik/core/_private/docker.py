from pathlib import Path
from typing import Any, Dict

import cloudtik

try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote

from cloudtik.core._private.cli_logger import cli_logger


def _check_docker_file_mounts(file_mounts: Dict[str, str]) -> None:
    """Checks if files are passed as file_mounts. This is a problem for Docker
    based clusters because when a file is bind-mounted in Docker, updates to
    the file on the host do not always propagate to the container. Using
    directories is recommended.
    """
    for remote, local in file_mounts.items():
        if Path(local).is_file():
            cli_logger.warning(
                f"File Mount: ({remote}:{local}) refers to a file.\n To ensure"
                " this mount updates properly, please use a directory.")


def validate_docker_config(config: Dict[str, Any]) -> None:
    """Checks whether the Docker configuration is valid."""
    if "docker" not in config:
        return

    docker_config = config["docker"]
    if docker_config.get("enabled", False):
        return

    _check_docker_file_mounts(config.get("file_mounts", {}))

    docker_image = docker_config.get("image")
    cname = docker_config.get("container_name")
    head_docker_image = docker_config.get("head_image", docker_image)
    worker_docker_image = docker_config.get("worker_image", docker_image)
    image_present = docker_image or (head_docker_image and worker_docker_image)
    
    assert cname and image_present, "Must provide a container & image name"


def with_docker_exec(cmds,
                     container_name,
                     docker_cmd,
                     env_vars=None,
                     with_interactive=False):
    assert docker_cmd, "Must provide docker command"
    env_str = ""
    if env_vars:
        env_str = " ".join(
            ["-e {env}=${env}".format(env=env) for env in env_vars])
    return [
        "{docker_cmd} exec {interactive} {env} {container} /bin/bash -c {cmd} ".
        format(
            docker_cmd=docker_cmd,
            interactive="-it" if with_interactive else "",
            env=env_str,
            container=container_name,
            cmd=quote(cmd)) for cmd in cmds
    ]


def get_docker_cmd(docker_cmd, with_sudo=False):
    return "{sudo}{docker_cmd}".format(
        sudo="sudo " if with_sudo else "",
        docker_cmd=docker_cmd)


def with_docker_cmd(cmd, docker_cmd, with_sudo=False):
    return "{docker_cmd} {cmd}".format(
        docker_cmd=get_docker_cmd(docker_cmd, with_sudo),
        cmd=cmd)


def _check_helper(cname, template, docker_cmd):
    return " ".join([
        docker_cmd, "inspect", "-f", "'{{" + template + "}}'", cname, "||",
        "true"
    ])


def check_docker_running_cmd(cname, docker_cmd):
    return _check_helper(cname, ".State.Running", docker_cmd)


def check_bind_mounts_cmd(cname, docker_cmd):
    return _check_helper(cname, "json .Mounts", docker_cmd)


def check_docker_image(cname, docker_cmd):
    return _check_helper(cname, ".Config.Image", docker_cmd)


def docker_start_cmds(user, image, mount_dict, data_disks, container_name, user_options,
                      cluster_name, home_directory, docker_cmd,
                      network=None, cpus=None, memory=None, labels=None,
                      port_mappings=None, mounts_mapping=False):
    mounts = mount_dict
    if mounts_mapping:
        # Imported here due to circular dependency.
        from cloudtik.core.api import get_docker_host_mount_location
        docker_mount_prefix = get_docker_host_mount_location(cluster_name)
        mounts = {dst: f"{docker_mount_prefix}/{dst}" for dst in mount_dict}

    return _docker_start_cmds(
        user, image, mounts, data_disks, container_name,
        user_options, home_directory, docker_cmd,
        network, cpus, memory, labels, port_mappings,
    )


def _docker_start_cmds(user, image, mounts, data_disks, container_name,
                       user_options, home_directory, docker_cmd,
                       network=None, cpus=None, memory=None, labels=None,
                       port_mappings=None):
    # mounts mapping: target -> source
    file_mounts = [
        "-v {src}:{dest}".format(
            src=v, dest=k.replace("~/", home_directory + "/"))
        for k, v in mounts.items()
    ]
    data_disk_mounts = [
        "-v {src}:{dest}".format(
            src=data_disk, dest=data_disk)
        for data_disk in data_disks
    ]

    volume_mounts = file_mounts
    volume_mounts += data_disk_mounts
    mount_flags = " ".join(volume_mounts)

    # for click, used in cloudtik cli
    env_vars = {"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
    env_flags = " ".join(
        ["-e {name}={val}".format(name=k, val=v) for k, v in env_vars.items()])

    user_options_str = " ".join(user_options)

    fuse_flags = "--cap-add SYS_ADMIN --device /dev/fuse --security-opt apparmor:unconfined"
    numactl_flag = "--cap-add SYS_NICE"
    ipc_flag = "--ipc=host"
    network_flag = "--network={}".format(network) if network else "--network=host"

    docker_run = [
        docker_cmd, "run", "--rm", "--name {}".format(container_name), "-d",
        "-it", mount_flags, env_flags, fuse_flags, user_options_str,
        numactl_flag, ipc_flag, network_flag
    ]
    if cpus:
        cpus_flag = "--cpus={}".format(cpus)
        docker_run += [cpus_flag]
    if memory:
        memory_flag = "--memory={}".format(memory)
        docker_run += [memory_flag]
    if labels:
        docker_run += ["--label {name}={val}".format(
            name=k, val=v) for k, v in labels.items()]
    if network and port_mappings:
        # host net doesn't need port mapping
        docker_run += ["-p {host_port}:{container_port}".format(
            host_port=host_port, container_port=container_port
        ) for container_port, host_port in port_mappings.items()]

    docker_run += [
        image, "bash"
    ]
    return " ".join(docker_run)


def get_configured_docker_image(docker_config, as_head):
    image = docker_config.get(
            f"{'head' if as_head else 'worker'}_image",
            docker_config.get("image"))

    is_gpu = docker_config.get("is_gpu", False)
    return get_versioned_image(image, is_gpu)


def get_versioned_image(image, is_gpu: bool = False):
    if not image:
        return image

    # check whether the image tag is specified
    # if image tag is not specified, the current CloudTik version as tag
    if image.find(":") >= 0:
        return image

    version = cloudtik.__version__
    image_template = "{}:{}"
    if is_gpu:
        image_template += "-gpu"
    return image_template.format(image, version)
