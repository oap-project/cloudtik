import yaml
import subprocess

cluster_config_file = "~/cloudtik_bootstrap_config.yaml"
template_spark_yaml = "/tmp/default-spark.yaml.template"
template_spark_yaml_url = "https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/confs/default-spark.yaml.template"
tpcx_ai_default_spark_yaml = "~/runtime/benchmark-tools/tpcx-ai/driver/config/default-spark.yaml"

if __name__ == '__main__':
    from cloudtik.runtime.spark.api import ThisSparkCluster

    cluster = ThisSparkCluster()
    cluster_info = cluster.get_info()
    worker_num = cluster_info['total-workers-ready']

    def download_template_spark_yaml():
        try:
            subprocess.check_call(f"rm -f {template_spark_yaml}", shell=True)
            subprocess.check_call(f"wget {template_spark_yaml_url} -O {template_spark_yaml}", shell=True)
        except Exception as e:
            print(f"Failed to download template spark yaml config file. The exception is {e}")
            exit(1)

    def replace_conf_value(conf_file, dict):
        with open(conf_file) as f:
            read = f.read()
        with open(conf_file, 'w') as f:
            for key, val in dict.items():
                read = read.replace(key, val)
            f.write(read)

    cluster_config = yaml.safe_load(open(cluster_config_file).read())

    yarn_container_resource = cluster_config['runtime']['spark']['yarn_container_resource']
    yarn_container_maximum_vcores = yarn_container_resource['yarn_container_maximum_vcores']
    yarn_container_maximum_memory = yarn_container_resource['yarn_container_maximum_memory']

    number_of_executors = worker_num
    spark_executor_memory = int(yarn_container_maximum_memory * 0.8)
    spark_executor_memoryOverhead = max(384, int(spark_executor_memory * 0.1))
    spark_driver_memory = cluster_config['runtime']['spark']['spark_executor_resource']['spark_driver_memory']

    executor_cores_horovod = max(1, int(yarn_container_maximum_vcores / 16))

    if worker_num < 2 and executor_cores_horovod == 1:
        executor_cores_horovod = 2

    dict = {'{%case02_executor_cores_horovod%}': str(executor_cores_horovod),
            '{%case05_executor_cores_horovod%}': str(executor_cores_horovod),
            '{%case09_executor_cores_horovod%}': str(executor_cores_horovod),
            '{%spark.executor.cores%}': str(yarn_container_maximum_vcores),
            '{%spark.executor.instances%}': str(number_of_executors),
            '{%spark.driver.memory%}': str(spark_driver_memory),
            '{%spark.executor.memory%}': str(spark_executor_memory),
            '{%spark.executor.memoryOverhead%}': str(spark_executor_memoryOverhead)}

    download_template_spark_yaml()
    replace_conf_value(template_spark_yaml, dict)

    try:
        subprocess.check_call(f"mv {template_spark_yaml} {tpcx_ai_default_spark_yaml}", shell=True)
        print("Successfully updated spark configuration for tpcx-ai")
    except Exception as e:
        print("Faild to update spark configuration for tpcx-ai")
