ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-runtime:"$BASE_IMAGE"

# Compile spark-sql-perf, tpcds-kit, tpch-dbgen, HiBench
RUN wget -P ~/ https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/spark/scripts/bootstrap-benchmark.sh -o /tmp/bootstrap-benchmark.sh \
    && bash /tmp/bootstrap-benchmark.sh --workload=all \
    && rm -rf /tmp/bootstrap-benchmark.sh
