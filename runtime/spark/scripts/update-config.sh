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
provider=$1
HEAD_ADDRESS=$2
if [ ! -n "${HEAD_ADDRESS}" ]; then
	local_host="`hostname --fqdn`"
	HEAD_ADDRESS=`nslookup -sil $local_host 2>/dev/null | grep Address: | sed '1d' | sed 's/Address: //g'`
	Is_head_node=true
else
	Is_head_node=false
fi

#For nodemanager
total_memory=$(awk '($1 == "MemTotal:"){print $2/1024}' /proc/meminfo)
total_memory=${total_memory%.*}
total_vcores=$(cat /proc/cpuinfo | grep processor | wc -l)

#For master
if [ $Is_head_node == "true" ];then
	memory_resource_Bytes=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."available_node_types"."worker.default"."resources"."memory"' | sed 's/\"//g')
	CPU_resource=$(cat ~/cloudtik_bootstrap_config.yaml | jq '."available_node_types"."worker.default"."resources"."CPU"' | sed 's/\"//g')
	memory_resource_MB=`expr $memory_resource_Bytes / 1048576`
	if [ $CPU_resource -lt 4 ]; then
		spark_executor_cores=$CPU_resource
		spark_executor_memory=${memory_resource_MB}M
	else
		spark_executor_cores=4
	        spark_executor_memory=8096M	
	fi
	spark_driver_memory=2048M
fi

cd $output_dir
sed -i "s/HEAD_ADDRESS/${HEAD_ADDRESS}/g" `grep "HEAD_ADDRESS" -rl ./`
sed -i "s!{%HADOOP_HOME%}!${HADOOP_HOME}!g" `grep "{%HADOOP_HOME%}" -rl ./`
sed -i "s/{%yarn.scheduler.maximum-allocation-mb%}/${total_memory}/g" `grep "{%yarn.scheduler.maximum-allocation-mb%}" -rl ./`
sed -i "s/{%yarn.nodemanager.resource.memory-mb%}/${total_memory}/g" `grep "{%yarn.nodemanager.resource.memory-mb%}" -rl ./`
sed -i "s/{%yarn.nodemanager.resource.cpu-vcores%}/${total_vcores}/g" `grep "{%yarn.nodemanager.resource.cpu-vcores%}" -rl ./`
sed -i "s/{%yarn.scheduler.maximum-allocation-vcores%}/${total_vcores}/g" `grep "{%yarn.scheduler.maximum-allocation-vcores%}" -rl ./`

if [ $Is_head_node == "true" ];then
	sed -i "s/{%spark.executor.cores%}/${spark_executor_cores}/g" `grep "{%spark.executor.cores%}" -rl ./`
	sed -i "s/{%spark.executor.memory%}/${spark_executor_memory}/g" `grep "{%spark.executor.memory%}" -rl ./`
	sed -i "s/{%spark.driver.memory%}/${spark_driver_memory}/g" `grep "{%spark.driver.memory%}" -rl ./`
fi

cp -r ${output_dir}/hadoop/${provider}/core-site.xml  ${HADOOP_HOME}/etc/hadoop/
cp -r ${output_dir}/hadoop/yarn-site.xml  ${HADOOP_HOME}/etc/hadoop/

if [ $Is_head_node == "true" ];then
	cp -r ${output_dir}/spark/*  ${SPARK_HOME}/conf
fi

jars=('spark-[0-9]*[0-9]-yarn-shuffle.jar' 'spark-network-common_[0-9]*[0-9].jar' 'spark-network-shuffle_[0-9]*[0-9].jar' 'jackson-databind-[0-9]*[0-9].jar' 'jackson-core-[0-9]*[0-9].jar' 'jackson-annotations-[0-9]*[0-9].jar' 'metrics-core-[0-9]*[0-9].jar' 'netty-all-[0-9]*[0-9].Final.jar' 'commons-lang3-[0-9]*[0-9].jar')
find ${HADOOP_HOME}/share/hadoop/yarn/lib -name netty-all-[0-9]*[0-9].Final.jar| xargs -i mv -f {} {}.old
for jar in ${jars[@]};
do
	find ${SPARK_HOME}/jars/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib; 
	find ${SPARK_HOME}/yarn/ -name $jar | xargs -i cp {} ${HADOOP_HOME}/share/hadoop/yarn/lib;
done

#Add share/hadoop/tools/lib/* into classpath 
echo "export HADOOP_CLASSPATH=\$HADOOP_CLASSPATH:\$HADOOP_HOME/share/hadoop/tools/lib/*" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh
