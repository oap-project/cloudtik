"""Tests launch, teardown, and update of multiple clusters using Kubernetes
operator. Also tests submission of jobs."""
import copy
import sys
import os
import subprocess
import tempfile
import time
import unittest

from contextlib import contextmanager

import kubernetes
import pytest
import yaml

import cloudtik

from cloudtik.providers._private._kubernetes.node_provider import KubernetesNodeProvider

IMAGE_ENV = "KUBERNETES_OPERATOR_TEST_IMAGE"
IMAGE = os.getenv(IMAGE_ENV, "cloudtik/cloudtik:nightly")

NAMESPACE_ENV = "KUBERNETES_OPERATOR_TEST_NAMESPACE"
NAMESPACE = os.getenv(NAMESPACE_ENV, "test-k8s-operator")

PULL_POLICY_ENV = "KUBERNETES_OPERATOR_TEST_PULL_POLICY"
PULL_POLICY = os.getenv(PULL_POLICY_ENV, "Always")

CLOUDTIK_PATH = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )))
)


@contextmanager
def client_connect_to_k8s(port="6789"):
    command = (
        f"kubectl -n {NAMESPACE}"
        f" port-forward service/cloudtik-example-cluster-head {port}:{port}"
    )
    command = command.split()
    print(">>>Port-forwarding head service.")
    proc = subprocess.Popen(command)
    # Wait a bit for the port-forwarding connection to be
    # established.
    time.sleep(10)
    try:
        yield proc
    finally:
        proc.kill()


def retry_until_true(f):
    # Keep retrying for 8 minutes with 10 seconds between attempts.
    def f_with_retries(*args, **kwargs):
        for _ in range(49):
            if f(*args, **kwargs):
                return
            else:
                time.sleep(10)
        pytest.fail("The condition wasn't met before the timeout expired.")

    return f_with_retries


@retry_until_true
def wait_for_pods(n, namespace=NAMESPACE, name_filter=""):
    client = kubernetes.client.CoreV1Api()
    pods = client.list_namespaced_pod(namespace=namespace).items
    count = 0
    for pod in pods:
        if name_filter in pod.metadata.name:
            count += 1
            # Double-check that the correct image is use.
            assert pod.spec.containers[0].image == IMAGE, pod.spec.containers[0].image
    return count == n


@retry_until_true
def wait_for_logs(operator_pod):
    """Check if logs indicate presence of nodes of types "head-node" and
    "worker-nodes" in the "example-cluster" cluster."""
    cmd = (
        f"kubectl -n {NAMESPACE} logs {operator_pod}"
        f"| grep ^example-cluster,{NAMESPACE}: | tail -n 100"
    )
    log_tail = subprocess.check_output(cmd, shell=True).decode()
    return ("head-node" in log_tail) and ("worker-node" in log_tail)


@retry_until_true
def wait_for_job(job_pod):
    print(">>>Checking job logs.")
    cmd = f"kubectl -n {NAMESPACE} logs {job_pod}"
    try:
        out = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        print(">>>Failed to check job logs.")
        print(e.output.decode())
        return False
    success = "success" in out.lower()
    if success:
        print(">>>Job submission succeeded.")
    else:
        print(">>>Job logs do not indicate job sucess:")
        print(out)
    return success


@retry_until_true
def wait_for_command_to_succeed(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


@retry_until_true
def wait_for_command_to_succeed_on_head(cmd_template, head_filter, namespace):
    try:
        head_pod = [pod for pod in pods() if head_filter in pod].pop()
        wait_for_pod_status(head_pod, "Running")
        cmd = cmd_template.format(namespace=namespace, head_pod=head_pod)
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


@retry_until_true
def wait_for_pod_status(pod_name, status):
    client = kubernetes.client.CoreV1Api()
    pod = client.read_namespaced_pod(namespace=NAMESPACE, name=pod_name)
    return pod.status.phase == status


@retry_until_true
def wait_for_status(cluster_name, status):
    client = kubernetes.client.CustomObjectsApi()
    cluster_cr = client.get_namespaced_custom_object(
        namespace=NAMESPACE,
        group="cloudtik.io",
        version="v1",
        plural="cloudtikclusters",
        name=cluster_name,
    )
    return cluster_cr["status"]["phase"] == status


@retry_until_true
def wait_for_services(n):
    return num_services() == n


def kubernetes_tools_directory():
    relative_path = "tools/kubernetes"
    return os.path.join(CLOUDTIK_PATH, relative_path)


def get_kubernetes_tool_path(name):
    return os.path.join(kubernetes_tools_directory(), name)


def get_operator_config_path(file_name):
    operator_configs = get_kubernetes_tool_path("operator")
    return os.path.join(operator_configs, file_name)


def get_crd_path():
    return get_kubernetes_tool_path("helm/charts/cloudtik/crds/cluster_crd.yaml")


def pods():
    client = kubernetes.client.CoreV1Api()
    pod_items = client.list_namespaced_pod(namespace=NAMESPACE).items
    return [
        pod.metadata.name
        for pod in pod_items
        if pod.status.phase in ["Running", "Pending"]
        and pod.metadata.deletion_timestamp is None
    ]


def num_services():
    cmd = (
        f"kubectl -n {NAMESPACE} get services --no-headers -o"
        ' custom-columns=":metadata.name"'
    )
    service_list = subprocess.check_output(cmd, shell=True).decode().split()
    return len(service_list)


class KubernetesOperatorTest(unittest.TestCase):
    def test_basic(self):
        # Validate terminate_node error handling
        provider = KubernetesNodeProvider(
            {"namespace": NAMESPACE}, "default_cluster_name"
        )
        # 404 caught, no error
        provider.terminate_node("no-such-node")

        with tempfile.NamedTemporaryFile(
            "w+"
        ) as example_cluster_file, tempfile.NamedTemporaryFile(
            "w+"
        ) as example_cluster2_file, tempfile.NamedTemporaryFile(
            "w+"
        ) as operator_file, tempfile.NamedTemporaryFile(
            "w+"
        ) as job_file:

            # Get paths to operator configs
            example_cluster_config_path = get_operator_config_path(
                "example_cluster.yaml"
            )
            operator_config_path = get_operator_config_path("operator_namespaced.yaml")
            job_path = get_operator_config_path("example_job.yaml")

            # Load operator configs
            example_cluster_config = yaml.safe_load(
                open(example_cluster_config_path).read()
            )
            example_cluster2_config = copy.deepcopy(example_cluster_config)
            # One worker for the second config
            example_cluster2_config["spec"]["podTypes"][1]["minWorkers"] = 1
            example_cluster2_config["metadata"]["name"] = "example-cluster2"
            operator_config = list(
                yaml.safe_load_all(open(operator_config_path).read())
            )
            job_config = yaml.safe_load(open(job_path).read())

            # Fill image fields
            podTypes = example_cluster_config["spec"]["podTypes"]
            podTypes2 = example_cluster2_config["spec"]["podTypes"]
            pod_specs = (
                [operator_config[-1]["spec"]["template"]["spec"]]
                + [job_config["spec"]["template"]["spec"]]
                + [podType["podConfig"]["spec"] for podType in podTypes]
                + [podType["podConfig"]["spec"] for podType in podTypes2]
            )
            for pod_spec in pod_specs:
                pod_spec["containers"][0]["image"] = IMAGE
                pod_spec["containers"][0]["imagePullPolicy"] = PULL_POLICY

            # Dump to temporary files
            yaml.dump(example_cluster_config, example_cluster_file)
            yaml.dump(example_cluster2_config, example_cluster2_file)
            yaml.dump(job_config, job_file)
            yaml.dump_all(operator_config, operator_file)
            files = [example_cluster_file, example_cluster2_file, operator_file]
            for file in files:
                file.flush()

            # Start operator and two clusters
            print("\n>>>Starting operator and two clusters.")
            for file in files:
                cmd = f"kubectl -n {NAMESPACE} apply -f {file.name}"
                subprocess.check_call(cmd, shell=True)

            # Check that autoscaling respects minWorkers by waiting for
            # six pods in the namespace.
            print(">>>Waiting for pods to join clusters.")
            wait_for_pods(6)
            # Check that head services are present.
            print(">>>Checking that head services are present.")
            wait_for_services(2)

            # Check that logging output looks normal (two workers connected to
            # the cluster example-cluster.)
            operator_pod = [pod for pod in pods() if "operator" in pod].pop()
            wait_for_logs(operator_pod)

            print(
                ">>>Checking that client connection is uninterrupted by"
                " operator restart."
            )
            with client_connect_to_k8s():
                print(">>>Restarting operator pod.")
                cmd = f"kubectl -n {NAMESPACE} delete pod {operator_pod}"
                subprocess.check_call(cmd, shell=True)
                wait_for_pods(6)
                operator_pod = [pod for pod in pods() if "operator" in pod].pop()
                wait_for_pod_status(operator_pod, "Running")
                time.sleep(5)

            # Delete head node of the first cluster. Recovery logic should
            # allow the rest of the test to pass.
            print(">>>Deleting cluster's head to test recovery.")
            head_pod = [pod for pod in pods() if "cloudtik-example-cluster-head" in pod].pop()
            cd = f"kubectl -n {NAMESPACE} delete pod {head_pod}"
            subprocess.check_call(cd, shell=True)
            print(">>>Confirming recovery.")
            # Status marked "Running".
            wait_for_status("example-cluster", "Running")
            # Head pod recovered.
            wait_for_pods(6)

            # Delete the second cluster
            print(">>>Deleting example-cluster2.")
            cmd = f"kubectl -n {NAMESPACE} delete -f" f"{example_cluster2_file.name}"
            subprocess.check_call(cmd, shell=True)

            # Four pods remain
            print(">>>Checking that example-cluster2 pods are gone.")
            wait_for_pods(4)
            # Cluster 2 service has been garbage-collected.
            print(">>>Checking that deleted cluster's service is gone.")
            wait_for_services(1)

            # Check job submission
            print(">>>Submitting a job to test.")
            cmd = f"kubectl -n {NAMESPACE} create -f {job_file.name}"
            subprocess.check_call(cmd, shell=True)
            wait_for_pods(1, name_filter="job")
            job_pod = [pod for pod in pods() if "job" in pod].pop()
            time.sleep(10)
            wait_for_job(job_pod)
            cmd = f"kubectl -n {NAMESPACE} delete jobs --all"
            subprocess.check_call(cmd, shell=True)

            # Check that cluster updates work: increase minWorkers to 3
            # and check that one worker is created.
            print(">>>Updating cluster size.")
            example_cluster_edit = copy.deepcopy(example_cluster_config)
            example_cluster_edit["spec"]["podTypes"][1]["minWorkers"] = 3
            yaml.dump(example_cluster_edit, example_cluster_file)
            example_cluster_file.flush()
            cm = f"kubectl -n {NAMESPACE} apply -f {example_cluster_file.name}"
            subprocess.check_call(cm, shell=True)
            print(">>>Checking that new cluster size is respected.")
            wait_for_pods(5)

            # Delete the first cluster
            print(">>>Deleting second cluster.")
            cmd = f"kubectl -n {NAMESPACE} delete -f" f"{example_cluster_file.name}"
            subprocess.check_call(cmd, shell=True)

            # Only operator pod remains.
            print(">>>Checking that all cluster pods are gone.")
            wait_for_pods(1)

            # Cluster 1 service has been garbage-collected.
            print(">>>Checking that all cluster services are gone.")
            wait_for_services(0)

            # Verify that cluster deletion earlier in this test did not break
            # the operator.
            print(">>>Checking cluster creation again.")
            for file in [example_cluster_file, example_cluster2_file]:
                cmd = f"kubectl -n {NAMESPACE} apply -f {file.name}"
                subprocess.check_call(cmd, shell=True)
            wait_for_pods(7)
            print(">>>Checking cluster deletion again.")
            for file in [example_cluster_file, example_cluster2_file]:
                cmd = f"kubectl -n {NAMESPACE} delete -f {file.name}"
                subprocess.check_call(cmd, shell=True)
            wait_for_pods(1)


if __name__ == "__main__":
    kubernetes.config.load_kube_config()
    sys.exit(pytest.main(["-sv", __file__]))
