#!/bin/bash

if [ -z "TRINO_HOME" ]; then
    echo "Please make sure that you've installed trino runtime!"
    exit 1
fi

args=$(getopt -a -o i::s::w::h:: -l iteration::,scale::,workload::,help:: -- "$@")
eval set -- "${args}"

ITERATION=1
SCALE=1
WORKLOAD=tpcds

function usage() {
    echo "Usage: $0 -i=[iteration] -s=[scale] -w=[workload] | -h | --help" >&2
}

function prepare_coordinator() {
    echo "Getting coordinator..."
    head_ip=$(cloudtik head head-ip)
    echo "Successfully get coordinator: ${head_ip}"
}

function contains() {
    local n=$#
    local value=${!n}
    for ((i=1;i < $#;i++)) {
        if [ "${!i}" == "${value}" ]; then
            echo "y"
            return 0
        fi
    }
    echo "n"
    return 1
}

function check_data_scale(){
    if [ "${WORKLOAD}" == "tpch" ];then
        TPCH_ALLOWED_SF=(1 100 1000 10000 100000 300 3000 30000)
        if [ $(contains "${TPCH_ALLOWED_SF[@]}" "$SCALE") == "y" ]; then
            echo "SF$SCALE is allowed for TPCH."
        else
            echo "SF$SCALE is not allowed for TPCH. Supported SF: ${TPCH_ALLOWED_SF[*]}."
            exit 1
        fi
    elif [ "${WORKLOAD}" == "tpcds" ];then
        TPCDS_ALLOWED_SF=(1 10 100 1000 10000 100000 300 3000 30000)
        if [ $(contains "${TPCDS_ALLOWED_SF[@]}" "$SCALE") == "y" ]; then
            echo "SF$SCALE is allowed for TPCDS."
        else
            echo "SF$SCALE is not allowed for TPCDS. Supported SF: ${TPCDS_ALLOWED_SF[*]}."
            exit 1
        fi
    fi
}

function prepare_tpc_connector(){
    tpc_workload=$1
    cloudtik head exec --all-nodes --no-parallel "mkdir -p \${TRINO_HOME}/etc/catalog/"
    cloudtik head exec --all-nodes --no-parallel "echo connector.name=${tpc_workload} > \${TRINO_HOME}/etc/catalog/${tpc_workload}.properties"
    cloudtik head runtime start --runtimes trino -y
    sleep 30
}

function prepare_tpc_queries(){
    tpc_workload=$1
    if [ ! -d "/tmp/repo/trino" ]; then
        mkdir -p /tmp/repo
        git clone https://github.com/trinodb/trino.git /tmp/repo/trino
    fi
    rm -rf $TRINO_HOME/${tpc_workload}/${tpc_workload}-queries && mkdir -p $TRINO_HOME/${tpc_workload}
    cp -r /tmp/repo/trino/testing/trino-benchto-benchmarks/src/main/resources/sql/presto/${tpc_workload} $TRINO_HOME/${tpc_workload}/${tpc_workload}-queries
    if [ "${tpc_workload}" == "tpcds" ]; then
        for query in $TRINO_HOME/tpcds/tpcds-queries/*
        do
            echo ";" >> "$query"
        done
    fi
    database=tpcds
    schema=${SCALE}
    prefix=""
    sed -i "s#\${database}#${database}#g" `grep '\${database}' -rl $TRINO_HOME/${tpc_workload}`
    sed -i "s#\${schema}#sf${schema}#g" `grep '\${schema}' -rl $TRINO_HOME/${tpc_workload}`
    sed -i "s#\${prefix}#${prefix}#g" `grep '\${prefix}' -rl $TRINO_HOME/${tpc_workload}`
}

function run_tpch(){
    log_dir=$TRINO_HOME/tpch/SF${SCALE}
    rm -rf ${log_dir}
    mkdir -p ${log_dir}
    queries_dir=$TRINO_HOME/tpch/tpch-queries
    for round in $(seq $ITERATION);do
        echo "Runing $round round"!
        log_current_dir=${log_dir}/$round
        mkdir ${log_current_dir}
        for query in `seq 1 22`; do
            query=`printf "%02d" $query`
            start=$(date +%s)
            trino --server ${head_ip}:8080 --file $queries_dir/q${query}.sql --output-format ALIGNED > ${log_current_dir}/q${query}.log
            if [ $? -eq 0 ];then
                RES=Success
            else
                RES=Fail
            fi
            end=$(date +%s)
            time=$(( $end - $start ))
            echo "q${query} $time $RES" >> ${log_current_dir}/result.log
            echo "q${query},$time,$RES" >> ${log_current_dir}/result.csv
        done
        echo "The final result directory is: ${log_current_dir}"
    done
}

function run_tpcds(){
    log_dir=$TRINO_HOME/tpcds/SF${SCALE}
    rm -rf ${log_dir}
    mkdir -p ${log_dir}
    queries_dir=$TRINO_HOME/tpcds/tpcds-queries
    for round in $(seq $ITERATION);do
        echo "Runing $round round"!
        log_current_dir=${log_dir}/$round
        mkdir ${log_current_dir}
        for query in `seq 1 99`; do
            query=`printf "%02d" $query`

            if [ -e "${queries_dir}/q${query}.sql" ]; then
                start=$(date +%s)
                trino --server ${head_ip}:8080 --file ${queries_dir}/q${query}.sql --output-format ALIGNED > ${log_current_dir}/q${query}.log
                if [ $? -eq 0 ];then
                    RES=Success
                else
                    RES=Fail
                fi
                end=$(date +%s)
                time=$(( $end - $start ))
                echo "q${t} $time $RES" >> ${log_current_dir}/result.log
                echo "q${t},$time,$RES" >> ${log_current_dir}/result.csv
            fi

            if [ -e "${queries_dir}/q${query}_1.sql" ]; then
                start=$(date +%s)
                trino --server ${head_ip}:8080 --file ${queries_dir}/q${query}_1.sql --output-format ALIGNED > ${log_current_dir}/q${query}_1.log
                if [ $? -eq 0 ];then
                    RES=Success
                else
                    RES=Fail
                fi
                end=$(date +%s)
                time=$(( $end - $start ))
                echo "q${t}_1 $time $RES" >> ${log_current_dir}/result.log
                echo "q${t}_1,$time,$RES" >> ${log_current_dir}/result.csv
            fi

            if [ -e "${queries_dir}/q${query}_2.sql" ]; then
                start=$(date +%s)
                trino --server ${head_ip}:8080 --file ${queries_dir}/q${query}_2.sql --output-format ALIGNED > ${log_current_dir}/q${query}_2.log
                if [ $? -eq 0 ];then
                    RES=Success
                else
                    RES=Fail
                fi
                end=$(date +%s)
                time=$(( $end - $start ))
                echo "q${t}_2 $time $RES" >> ${log_current_dir}/result.log
                echo "q${t}_2,$time,$RES" >> ${log_current_dir}/result.csv
            fi
        done
        echo "The final result directory is: ${log_current_dir}"
    done
}

while true
do
    case "$1" in
    -i|--iteration)
        ITERATION=$2
        ;;
    -s|--scale)
        SCALE=$2
        shift
        ;;
    -w|--workload)
        WORKLOAD=$2
        shift
        ;;
    -h|--help)
        shift
        usage
        exit 0
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done


prepare_coordinator
check_data_scale

if [ "${WORKLOAD}" == "tpcds" ];then
    prepare_tpc_connector tpcds
    prepare_tpc_queries tpcds
    run_tpcds
elif [ "${WORKLOAD}" == "tpch" ];then
    prepare_tpc_connector tpch
    prepare_tpc_queries tpch
    run_tpch
else
    echo "Only support tpcds and tpch workload."
    exit 1
fi
