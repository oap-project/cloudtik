import pytest
import sys
import os
import json
import redis

EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
CLOUDTIK_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
CLOUDTIK_REDIS_EXECUTABLE = os.path.join(
    CLOUDTIK_PATH, "core/thirdparty/redis/redis-server" + EXE_SUFFIX)

from cloudtik.core._private.services import start_cloudtik_process
import cloudtik.core._private.constants as constants
from cloudtik.core._private.state.control_state import ControlState
import time

processes = []
TEST_KEYS = ['node-1', 'node-2', 'node-3', 'node-4', 'node-5']


def setup_module():
    redis_executable = CLOUDTIK_REDIS_EXECUTABLE
    print(redis_executable)
    command = [redis_executable, '--requirepass', constants.CLOUDTIK_REDIS_DEFAULT_PASSWORD, '--port']

    for port in [constants.CLOUDTIK_DEFAULT_PORT, '52345']:
        cmd = command.copy()
        cmd.append(str(port))
        print(cmd)
        process_info = start_cloudtik_process(
            cmd,
            constants.PROCESS_TYPE_REDIS_SERVER,
            fate_share=False)
        print(process_info)
        processes.append(process_info)
    time.sleep(10)

def teardown_module():
    print("teardown_function--->")
    for process_info in processes:
        process = process_info.process
        process.terminate()
        process.wait()
        if process.returncode != 0:
            message = ("Valgrind detected some errors in process of "
                       "type {}. Error code {}.".format(
                constants.PROCESS_TYPE_REDIS_SERVER, process.returncode))
            if process_info.stdout_file is not None:
                with open(process_info.stdout_file, "r") as f:
                    message += "\nPROCESS STDOUT:\n" + f.read()
            if process_info.stderr_file is not None:
                with open(process_info.stderr_file, "r") as f:
                    message += "\nPROCESS STDERR:\n" + f.read()
            raise RuntimeError(message)


class TestNodeTable:
    @classmethod
    def setup_class(self):
        redis_client = redis.StrictRedis(host='127.0.0.1', port=constants.CLOUDTIK_DEFAULT_PORT,
                                         password=constants.CLOUDTIK_REDIS_DEFAULT_PASSWORD)
        redis_client.rpush("RedisShards", "127.0.0.1:52345")
        redis_client.set("NumRedisShards", 1)
        self.control_state = ControlState()
        self.control_state.initialize_control_state('127.0.0.1', constants.CLOUDTIK_DEFAULT_PORT,
                                                    constants.CLOUDTIK_REDIS_DEFAULT_PASSWORD)
        self.node_table = self.control_state.get_node_table()

    @pytest.mark.parametrize("key", TEST_KEYS)
    @pytest.mark.parametrize("value",
                             [{"ip": "127.0.0.1"}, {"ip": "127.0.0.2"}, {"ip": "127.0.0.3"}, {"ip": "127.0.0.4"}])
    def test_put(self, key, value):
        value_json = json.dumps(value)

        self.node_table.put(key, value_json)

    def test_get_all(self):
        res = self.node_table.get_all()
        for key in TEST_KEYS:
            assert key in res.keys()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
