#!/bin/bash

if [ ! -n "${HADOOP_HOME}" ]; then
  echo "HADOOP_HOME environment variable is not set."
  exit 1
fi

if [ ! -n "${SPARK_HOME}" ]; then
  echo "SPARK_HOME environment variable is not set."
  exit 1
fi

source_dir=$(cd $(dirname ${BASH_SOURCE[0]})/..;pwd)/conf
output_dir=$(dirname ${source_dir})/outconf

rm -rf  $output_dir
mkdir -p $output_dir
cp -r $source_dir/* $output_dir

master_hostname=$1
if [ ! -n "${master_hostname}" ]; then
	master_hostname=$(hostname) 
fi   

total_memory=$(awk '($1 == "MemTotal:"){print $2/1024}' /proc/meminfo)
total_vcores=$(cat /proc/cpuinfo | grep processor | wc -l)

spark_executor_cores=2
spark_executor_instances=2
spark_executor_memory=1g
spark_yarn_executor_memoryOverhead=384

cd $output_dir
sed -i "s/master_hostname/${master_hostname}/g" `grep "master_hostname" -rl ./`
sed -i "s/{%HADOOP_HOME%}/${HADOOP_HOME}/g" `grep "{%HADOOP_HOME%}" -rl ./`
sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`

sed -i "s/{%spark.executor.cores%}/${spark_executor_cores}/g" `grep "{%spark.executor.cores%}" -rl ./`
sed -i "s/{%spark.executor.instances%}/${spark_executor_instances}/g" `grep "{%spark.executor.instances%}" -rl ./`
sed -i "s/{%spark.executor.memory%}/${spark_executor_memory}/g" `grep "{%spark.executor.memory%}" -rl ./`
sed -i "s/{%spark.yarn.executor.memoryOverhead%}/${spark_yarn_executor_memoryOverhead}/g" `grep "{%spark.yarn.executor.memoryOverhead%}" -rl ./`

cp -r ${output_dir}/hadoop/*  ${HADOOP_HOME}/etc/hadoop/
cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf/spark/
