ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ml-runtime:"$BASE_IMAGE"

# Deploy TPCx-AI toolkit
RUN wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/scripts/bootstrap-tpcx-ai-benchmark.sh -O /tmp/bootstrap-tpcx-ai-benchmark.sh \
    && bash /tmp/bootstrap-tpcx-ai-benchmark.sh --head \
    && rm -rf /tmp/bootstrap-tpcx-ai-benchmark.sh
