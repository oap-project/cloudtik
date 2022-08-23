import copy

import pytest

from cloudtik.core._private.config_utils import update_nested_dict, process_config_with_privacy, encrypt_config, \
    decrypt_config, hash_runtime_conf

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
