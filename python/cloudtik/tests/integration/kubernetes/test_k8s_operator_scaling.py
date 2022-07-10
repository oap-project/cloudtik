"""
Tests scaling behavior of Kubernetes operator.

(1) Start a cluster with minWorkers = n and verify scale-up
(2) Edit minWorkers to 0, verify scale-down
"""
import copy
import kubernetes
import subprocess
import sys
import tempfile
import time
import unittest

import pytest
import yaml

from test_k8s_operator_basic import get_crd_path
from test_k8s_operator_basic import get_operator_config_path
from test_k8s_operator_basic import retry_until_true
from test_k8s_operator_basic import wait_for_pods
from test_k8s_operator_basic import IMAGE
from test_k8s_operator_basic import PULL_POLICY
from test_k8s_operator_basic import NAMESPACE

TEST_SCALE_MIN_WORKERS = 3
TEST_SCALE_MAX_WORKERS = 10


@retry_until_true
def wait_for_operator():
    cmd = "kubectl get pods"
    out = subprocess.check_output(cmd, shell=True).decode()
    for line in out.splitlines():
        if "cloudtik-operator" in line and "Running" in line:
            return True
    return False


class KubernetesScaleTest(unittest.TestCase):
    def test_scaling(self):
        with tempfile.NamedTemporaryFile(
            "w+"
        ) as example_cluster_file, tempfile.NamedTemporaryFile(
            "w+"
        ) as example_cluster_file2, tempfile.NamedTemporaryFile(
            "w+"
        ) as operator_file:

            example_cluster_config_path = get_operator_config_path(
                "example_cluster.yaml"
            )
            operator_config_path = get_operator_config_path(
                "operator_cluster_scoped.yaml"
            )

            operator_config = list(
                yaml.safe_load_all(open(operator_config_path).read())
            )
            example_cluster_config = yaml.safe_load(
                open(example_cluster_config_path).read()
            )

            # Set image and pull policy
            podTypes = example_cluster_config["spec"]["podTypes"]
            pod_specs = [operator_config[-1]["spec"]["template"]["spec"]] + [
                podType["podConfig"]["spec"] for podType in podTypes
            ]
            for pod_spec in pod_specs:
                pod_spec["containers"][0]["image"] = IMAGE
                pod_spec["containers"][0]["imagePullPolicy"] = PULL_POLICY

            # Config set-up for this test.
            example_cluster_config["spec"]["maxWorkers"] = 100
            example_cluster_config["spec"]["idleTimeoutMinutes"] = 1
            worker_type = podTypes[1]
            # Make sure we have the right type
            assert "worker" in worker_type["name"]
            worker_type["maxWorkers"] = TEST_SCALE_MAX_WORKERS
            # Key for the first part of this test:
            worker_type["minWorkers"] = TEST_SCALE_MIN_WORKERS

            # Config for a small cluster with the same name to be launched
            # in another namespace.
            example_cluster_config2 = copy.deepcopy(example_cluster_config)
            example_cluster_config2["spec"]["podTypes"][1]["minWorkers"] = 1

            # Test overriding default client port.
            example_cluster_config["spec"]["headServicePorts"] = [
                {"name": "client", "port": 10002, "targetPort": 10001}
            ]

            yaml.dump(example_cluster_config, example_cluster_file)
            yaml.dump(example_cluster_config2, example_cluster_file2)
            yaml.dump_all(operator_config, operator_file)

            files = [example_cluster_file, operator_file]
            for file in files:
                file.flush()

            # Must create CRD before operator.
            print("\n>>>Creating CloudTikCluster CRD.")
            cmd = f"kubectl apply -f {get_crd_path()}"
            subprocess.check_call(cmd, shell=True)
            # Takes a bit of time for CRD to register.
            time.sleep(10)

            print(">>>Creating operator.")
            cmd = f"kubectl apply -f {operator_file.name}"
            subprocess.check_call(cmd, shell=True)

            print(">>>Waiting for CloudTik operator to enter running state.")
            wait_for_operator()

            # Start an n-pod cluster.
            print(">>>Starting a cluster.")
            cd = f"kubectl -n {NAMESPACE} apply -f {example_cluster_file.name}"
            subprocess.check_call(cd, shell=True)

            print(">>>Starting a cluster with same name in another namespace")
            # Assumes a namespace called {NAMESPACE}2 has been created.
            cd = f"kubectl -n {NAMESPACE}2 apply -f " f"{example_cluster_file2.name}"
            subprocess.check_call(cd, shell=True)

            # Check that autoscaling respects minWorkers by waiting for
            # 32 pods in one namespace and 2 pods in the other.
            print(">>>Waiting for pods to join cluster.")
            wait_for_pods(TEST_SCALE_MIN_WORKERS + 1)
            wait_for_pods(2, namespace=f"{NAMESPACE}2")

            # Check scale-down.
            print(">>>Decreasing min workers to 0.")
            example_cluster_edit = copy.deepcopy(example_cluster_config)
            # Set minWorkers to 0:
            example_cluster_edit["spec"]["podTypes"][1]["minWorkers"] = 0
            yaml.dump(example_cluster_edit, example_cluster_file)
            example_cluster_file.flush()
            cm = f"kubectl -n {NAMESPACE} apply -f {example_cluster_file.name}"
            subprocess.check_call(cm, shell=True)
            print(">>>Sleeping for a minute while workers time-out.")
            time.sleep(60)
            print(">>>Verifying scale-down.")
            wait_for_pods(1)


if __name__ == "__main__":
    kubernetes.config.load_kube_config()
    sys.exit(pytest.main(["-sv", __file__]))
