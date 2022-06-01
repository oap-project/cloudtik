#!/bin/bash

if [ -z "KAFKA_HOME" ]; then
    echo "Please make sure that you've installed kafka runtime!"
    exit 1
fi

function prepare_brokers() {
    echo "Getting brokers..."
    worker_ips=$(cloudtik  worker-ips ~/cloudtik_bootstrap_config.yaml)
    bootstrap_servers=""
    for worker_ip in ${worker_ips}; do
        if [ -z "$bootstrap_servers" ]; then
            bootstrap_servers=$worker_ip:9092
        else
            bootstrap_servers="$bootstrap_servers,$worker_ip:9092"
        fi
    done
    echo "Successfully get brokers: ${bootstrap_servers}"
}

function prepare_kafka_topic() {
    # delete the topic if exists
    echo "Deleting the topic cloudtik-kafka-benchmark if it exists..."
    $KAFKA_HOME/bin/kafka-topics.sh \
        --bootstrap-server ${bootstrap_servers} \
        --delete \
        --topic cloudtik-kafka-benchmark \
        --if-exists
    echo "Successfully delete the topic cloudtik-kafka-benchmark."

    # create the topic
    echo "Creating the topic cloudtik-kafka-benchmark..."
    $KAFKA_HOME/bin/kafka-topics.sh \
        --bootstrap-server ${bootstrap_servers} \
        --create \
        --topic cloudtik-kafka-benchmark \
        --partitions 6 --replication-factor 3
}

function producer_perf_test(){
    echo "Testing kafka cluster with single producer..."
    $KAFKA_HOME/bin/kafka-producer-perf-test.sh \
        --topic cloudtik-kafka-benchmark \
        --num-records 50000000 \
        --record-size 100 \
        --throughput -1 \
        --producer-props acks=1 \
        bootstrap.servers=${bootstrap_servers} \
        buffer.memory=67108864 \
        batch.size=8096
    echo "Testing kafka cluster with single producer has finished."
    echo ""

    echo "Testing producer with different record size..."
    for i in 10 100 1000 10000 100000; do
        echo ""
        echo "record_size=$i"
        $KAFKA_HOME/bin/kafka-producer-perf-test.sh \
          --topic cloudtik-kafka-benchmark \
          --num-records $((1000*1024*1024/$i))\
          --record-size $i\
          --throughput -1 \
          --producer-props acks=1 \
          bootstrap.servers=${bootstrap_servers} \
          buffer.memory=67108864 \
          batch.size=128000
    done;
    echo "Testing producer with different record size has finished."
}

function consumer_perf_test(){
    echo "Testing the consumer throughput of Kafka cluster..."
    $KAFKA_HOME/bin/kafka-consumer-perf-test.sh \
        --bootstrap-server ${bootstrap_servers} \
        --messages 50000000 \
        --topic cloudtik-kafka-benchmark
    echo "Testing the consumer throughput has finished..."
}

function end_to_end_latency(){
    echo "Testing end-to-end latency..."
    echo "num_msg Latency/ms"
    $KAFKA_HOME/bin/kafka-run-class.sh \
        kafka.tools.EndToEndLatency \
        ${bootstrap_servers} \
        cloudtik-kafka-benchmark 50000 1 1024
    echo "Testing end-to-end latency has finished."
}


prepare_brokers
prepare_kafka_topic
producer_perf_test
consumer_perf_test
end_to_end_latency
