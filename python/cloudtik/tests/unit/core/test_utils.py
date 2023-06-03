import copy

import pytest

from cloudtik.core._private.call_context import CallContext
from cloudtik.core._private.core_utils import get_memory_in_bytes, format_memory
from cloudtik.core._private.utils import update_nested_dict, process_config_with_privacy, encrypt_config, \
    decrypt_config, hash_runtime_conf, run_in_parallel_on_nodes, ParallelTaskSkipped, parse_resource_list, \
    parse_resources_json, parse_bundles_json, get_resource_list_str

TARGET_DICT_WITH_MATCHED_LIST = {
    "test_list": [
        {
            "name": "child-1",
            "value": 0,
            "test_value": 0
        }
    ]
}

NEW_DICT_WITH_MATCHED_LIST = {
    "test_list": [
        {
            "name": "child-1",
            "value": 1
        }
    ]
}

NEW_DICT_WITHOUT_MATCHED_LIST_1 = {
    "test_list": [
        {
            "name": "child-2",
            "value": 2
        }
    ]
}

NEW_DICT_WITHOUT_MATCHED_LIST_2 = {
    "test_list": [
        {
            "name": "child-1",
            "value": 1
        },
        {
            "name": "child-2",
            "value": 2
        }
    ]
}

NEW_DICT_WITHOUT_MATCHED_LIST_3 = {
    "test_list": [
        "child-1"
    ]
}


TARGET_DICT_APPENDING_LIST = {
    "test_list": [
        "b",
    ],
    "test_list_1": [
        "b1",
    ]
}

NEW_DICT_APPENDING_LIST = {
    "++test_list": [
        "a",
    ],
    "test_list++": [
        "c",
    ],
    "++test_list_1": [
        "a1",
    ],
    "test_list_1++": [
        "c1",
    ]
}

NEW_DICT_APPENDING_LIST_1 = {
    "++test_list": [
        "a",
    ],
    "test_list": [
        "B",
    ],
    "test_list++": [
        "c",
    ],
    "++test_list_1": [
        "a1",
    ],
    "test_list_1": [
        "B1",
    ],
    "test_list_1++": [
        "c1",
    ]
}

TEST_CONFIG_WITH_PRIVACY = {
    "test_array": [
        {
            "no_privacy": "abc",
            "account.key": "123",
        }
    ],
    "test_dict": {
        "no_privacy": "abc",
        "Account.Key": "123",
        "nested_dict": {
            "no_privacy": "abc",
            "credentials": "123"
        }
    }
}


class TestUtils:
    def test_updated_nested_dict_match_list(self):
        updated_dict = update_nested_dict(
            copy.deepcopy(TARGET_DICT_WITH_MATCHED_LIST),
            copy.deepcopy(NEW_DICT_WITH_MATCHED_LIST),
            match_list_item_with_name=True)

        assert len(updated_dict["test_list"]) == 1
        assert updated_dict["test_list"][0]["name"] == "child-1"
        assert updated_dict["test_list"][0]["value"] == 1
        assert updated_dict["test_list"][0]["test_value"] == 0

        updated_dict = update_nested_dict(
            copy.deepcopy(TARGET_DICT_WITH_MATCHED_LIST),
            copy.deepcopy(NEW_DICT_WITHOUT_MATCHED_LIST_1),
            match_list_item_with_name=True)

        assert len(updated_dict["test_list"]) == 1
        assert updated_dict["test_list"][0]["name"] == "child-2"
        assert updated_dict["test_list"][0]["value"] == 2
        assert "test_value" not in updated_dict["test_list"][0]

        updated_dict = update_nested_dict(
            copy.deepcopy(TARGET_DICT_WITH_MATCHED_LIST),
            copy.deepcopy(NEW_DICT_WITHOUT_MATCHED_LIST_2),
            match_list_item_with_name=True)

        assert len(updated_dict["test_list"]) == 2
        assert updated_dict["test_list"][0]["name"] == "child-1"
        assert updated_dict["test_list"][0]["value"] == 1
        assert "test_value" not in updated_dict["test_list"][0]

        updated_dict = update_nested_dict(
            copy.deepcopy(TARGET_DICT_WITH_MATCHED_LIST),
            copy.deepcopy(NEW_DICT_WITHOUT_MATCHED_LIST_3),
            match_list_item_with_name=True)

        assert len(updated_dict["test_list"]) == 1
        assert updated_dict["test_list"][0] == "child-1"

    def test_updated_nested_dict_advanced_list_appending(self):
        updated_dict = update_nested_dict(
            copy.deepcopy(TARGET_DICT_APPENDING_LIST),
            copy.deepcopy(NEW_DICT_APPENDING_LIST),
            match_list_item_with_name=True,
            advanced_list_appending=True)

        assert len(updated_dict["test_list"]) == 3
        assert updated_dict["test_list"][0] == "a"
        assert updated_dict["test_list"][1] == "b"
        assert updated_dict["test_list"][2] == "c"
        assert len(updated_dict["test_list_1"]) == 3
        assert updated_dict["test_list_1"][0] == "a1"
        assert updated_dict["test_list_1"][1] == "b1"
        assert updated_dict["test_list_1"][2] == "c1"

        updated_dict = update_nested_dict(
            copy.deepcopy(TARGET_DICT_APPENDING_LIST),
            copy.deepcopy(NEW_DICT_APPENDING_LIST_1),
            match_list_item_with_name=True,
            advanced_list_appending=True)

        assert len(updated_dict["test_list"]) == 3
        assert updated_dict["test_list"][0] == "a"
        assert updated_dict["test_list"][1] == "B"
        assert updated_dict["test_list"][2] == "c"
        assert len(updated_dict["test_list_1"]) == 3
        assert updated_dict["test_list_1"][0] == "a1"
        assert updated_dict["test_list_1"][1] == "B1"
        assert updated_dict["test_list_1"][2] == "c1"

    def test_process_config_with_privacy(self):
        config = copy.deepcopy(TEST_CONFIG_WITH_PRIVACY)
        process_config_with_privacy(config)

        # Make sure we don't see '123'
        # Make sure we saw 3 occurrence 'abc'
        assert config["test_array"][0]["no_privacy"] == "abc"
        assert config["test_array"][0]["account.key"] != "123"
        assert config["test_dict"]["no_privacy"] == "abc"
        assert config["test_dict"]["Account.Key"] != "123"
        assert config["test_dict"]["nested_dict"]["no_privacy"] == "abc"
        assert config["test_dict"]["nested_dict"]["credentials"] != "123"

    def test_encrypt_decrypt_config(self):
        config = copy.deepcopy(TEST_CONFIG_WITH_PRIVACY)
        secret_config = encrypt_config(config)
        assert secret_config["test_array"][0]["no_privacy"] == "abc"
        assert secret_config["test_array"][0]["account.key"] != "123"
        assert secret_config["test_dict"]["no_privacy"] == "abc"
        assert secret_config["test_dict"]["Account.Key"] != "123"
        assert secret_config["test_dict"]["nested_dict"]["no_privacy"] == "abc"
        assert secret_config["test_dict"]["nested_dict"]["credentials"] != "123"

        original_config = decrypt_config(secret_config)

        assert original_config["test_array"][0]["account.key"] == "123"
        assert original_config["test_dict"]["Account.Key"] == "123"
        assert original_config["test_dict"]["nested_dict"]["credentials"] == "123"

    def test_hash_runtime_conf(self):
        file_mounts = {}
        extra_objs = {"a": 1}
        config = {
            "available_node_types": {
                "head.default": {
                    "runtime": {
                        "a": 1
                    }
                },
                "worker.default": {
                    "runtime": {
                        "a": 1
                    },
                    "merged_commands": {
                        "worker_setup_commands": ["abc1"],
                        "worker_start_commands": ["xyz1"]
                    }
                }
            },
            "head_node_type": "head.default",
            "merged_commands": {
                "worker_setup_commands": ["abc"],
                "worker_start_commands": ["xyz"]
            }
        }

        (runtime_hash,
         file_mounts_contents_hash,
         runtime_hash_for_node_types) = hash_runtime_conf(
            file_mounts=file_mounts,
            cluster_synced_files=[],
            extra_objs=extra_objs,
            generate_file_mounts_contents_hash=True,
            generate_node_types_runtime_hash=True,
            config=config)

        assert runtime_hash is not None
        assert file_mounts_contents_hash is not None
        assert runtime_hash_for_node_types is not None
        assert len(runtime_hash_for_node_types) == 1

        (runtime_hash_1,
         file_mounts_contents_hash_1,
         runtime_hash_for_node_types_1) = hash_runtime_conf(
            file_mounts=file_mounts,
            cluster_synced_files=[],
            extra_objs=extra_objs,
            generate_file_mounts_contents_hash=False,
            generate_node_types_runtime_hash=False,
            config=config)

        assert runtime_hash_1 is not None
        assert file_mounts_contents_hash_1 is None
        assert runtime_hash_for_node_types_1 is None
        assert runtime_hash == runtime_hash_1

    def test_run_in_parallel_on_nodes(self):
        test_nodes = ["node-1", "node-2", "node-3"]
        call_context = CallContext()

        def task_success(node_id, call_context):
            pass

        succeeded, failures, skipped = run_in_parallel_on_nodes(
            task_success,
            call_context=call_context,
            nodes=test_nodes)
        assert succeeded == 3
        assert failures == 0
        assert skipped == 0

        def task_failed(node_id, call_context):
            raise RuntimeError("Task {} failed for no reason.".format(node_id))

        succeeded, failures, skipped = run_in_parallel_on_nodes(
            task_failed,
            call_context=call_context,
            nodes=test_nodes)
        assert succeeded == 0
        assert failures == 3
        assert skipped == 0

        def task_skipped(node_id, call_context):
            raise ParallelTaskSkipped("Task {} skipped for no reason.".format(node_id))

        succeeded, failures, skipped = run_in_parallel_on_nodes(
            task_skipped,
            call_context=call_context,
            nodes=test_nodes)
        assert succeeded == 0
        assert failures == 0
        assert skipped == 3

        def task_mix(node_id, call_context):
            if node_id == "node-1":
                pass
            elif node_id == "node-2":
                raise RuntimeError("Task {} failed for no reason.".format(node_id))
            elif node_id == "node-3":
                raise ParallelTaskSkipped("Task {} skipped for no reason.".format(node_id))

        succeeded, failures, skipped = run_in_parallel_on_nodes(
            task_mix,
            call_context=call_context,
            nodes=test_nodes)
        assert succeeded == 1
        assert failures == 1
        assert skipped == 1

    def test_memory_size_parsing(self):
        memory_size = None
        value = get_memory_in_bytes(memory_size)
        assert value == 0

        memory_size = 1024
        value = get_memory_in_bytes(memory_size)
        assert value == 1024

        memory_size = "4K"
        value = get_memory_in_bytes(memory_size)
        assert value == 4 * 1024

        memory_size = "4M"
        value = get_memory_in_bytes(memory_size)
        assert value == 4 * 1024 * 1024

        memory_size = "4G"
        value = get_memory_in_bytes(memory_size)
        assert value == 4 * 1024 * 1024 * 1024

    def test_resource_string(self):
        resource_list_str = 'CPU:4,GPU:1,Custom:3'
        resource_dict = parse_resource_list(resource_list_str)
        assert resource_dict["CPU"] == 4
        assert resource_dict["GPU"] == 1
        assert resource_dict["Custom"] == 3

        converted_str = get_resource_list_str(resource_dict)
        assert len(converted_str) == len(resource_list_str)
        assert "CPU:4" in converted_str
        assert "GPU:1" in converted_str
        assert "Custom:3" in converted_str

        resource_json = '{"CPU":4,"GPU":1,"Custom":3}'
        resource_dict = parse_resources_json(resource_json)
        assert resource_dict["CPU"] == 4
        assert resource_dict["GPU"] == 1
        assert resource_dict["Custom"] == 3

        bundles_json = '[{"CPU":4,"GPU":1,"Custom":3}, {"CPU":5,"GPU":2,"Custom":4}]'
        bundles = parse_bundles_json(bundles_json)
        assert len(bundles) == 2
        assert bundles[0]["CPU"] == 4
        assert bundles[0]["GPU"] == 1
        assert bundles[0]["Custom"] == 3
        assert bundles[1]["CPU"] == 5
        assert bundles[1]["GPU"] == 2
        assert bundles[1]["Custom"] == 4

    def test_format_memory(self):
        memory_in_bytes = 1024 * 1024
        memory_str = format_memory(memory_in_bytes)
        assert memory_str == "1MB"

        memory_in_bytes = 1024 * 1024 * 1024
        memory_str = format_memory(memory_in_bytes)
        assert memory_str == "1GB"

        memory_in_bytes = 1024 * 1024 * 1024 * 4.05
        memory_str = format_memory(memory_in_bytes)
        assert memory_str == "4.05GB"

        memory_in_bytes = 1024 * 1024 * 1024 * 1024
        memory_str = format_memory(memory_in_bytes)
        assert memory_str == "1TB"

        memory_in_bytes = 1024 * 1024 * 1024 * 1024 * 1.01
        memory_str = format_memory(memory_in_bytes)
        assert memory_str == "1.01TB"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
