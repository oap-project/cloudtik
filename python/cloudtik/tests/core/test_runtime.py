import os
import time

import pytest
from cloudtik.tests.core.constants import AWS_BASIC_CLUSTER_CONF_FILES, TPC_DATAGEN_BENCHMARK,  TPCDS_BENCHMARK, KAFKA_BENCHMARK



@pytest.mark.parametrize(
    'runtime_cluster_fixture',
    AWS_BASIC_CLUSTER_CONF_FILES,
    indirect=True
)
class TestClusterRuntime:

    @pytest.mark.parametrize("benchmark", [TPC_DATAGEN_BENCHMARK])
    def test_benchmark(self, runtime_cluster_fixture, benchmark):
        script_file = benchmark["script_file"]
        script_args = benchmark["script_args"]
        log_file_name = os.path.join("/tmp", "cloudtik_test_runtime_" + time.strftime("%Y-%m-%d", time.localtime(time.time())))
        runtime_cluster_fixture.submit(script_file=script_file, script_args=[script_args], log_file_name=log_file_name)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-vsx", __file__]))