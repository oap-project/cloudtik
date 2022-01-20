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
else
        master_hostname=ip-${master_hostname//./-}
fi

total_memory=$(awk '($1 == "MemTotal:"){print $2/1024}' /proc/meminfo)
total_memory=${total_memory%.*}
total_vcores=$(cat /proc/cpuinfo | grep processor | wc -l)

spark_executor_cores=1
spark_executor_memory=2g

cd $output_dir
sed -i "s/master_hostname/${master_hostname}/g" `grep "master_hostname" -rl ./`
sed -i "s!{%HADOOP_HOME%}!${HADOOP_HOME}!g" `grep "{%HADOOP_HOME%}" -rl ./`
sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`

sed -i "s/{%spark.executor.cores%}/${spark_executor_cores}/g" `grep "{%spark.executor.cores%}" -rl ./`
sed -i "s/{%spark.executor.memory%}/${spark_executor_memory}/g" `grep "{%spark.executor.memory%}" -rl ./`

cp -r ${output_dir}/hadoop/*  ${HADOOP_HOME}/etc/hadoop/
cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf

jars=('spark-[0-9]*[0-9]-yarn-shuffle.jar' 'spark-network-common_[0-9]*[0-9].jar' 'spark-network-shuffle_[0-9]*[0-9].jar' 'jackson-databind-[0-9]*[0-9].jar' 'jackson-core-[0-9]*[0-9].jar' 'jackson-annotations-[0-9]*[0-9].jar' 'metrics-core-[0-9]*[0-9].jar' 'netty-all-[0-9]*[0-9].Final.jar' 'commons-lang3-[0-9]*[0-9].jar')
find ${HADOOP_HOME}/share/hadoop/yarn/lib -name netty-all-[0-9]*[0-9].Final.jar| xargs -i mv -f {} {}.old
for jar in ${jars[@]};
do
	find ${SPARK_HOME}/jars/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib; 
	find ${SPARK_HOME}/yarn/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib;
done

