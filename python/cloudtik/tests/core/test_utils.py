import copy

import pytest

from cloudtik.core._private.utils import update_nested_dict

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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
