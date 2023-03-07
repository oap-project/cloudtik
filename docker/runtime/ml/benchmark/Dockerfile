ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ml-runtime:"$BASE_IMAGE"

ARG CLOUDTIK_REGION=""

RUN if [ "${CLOUDTIK_REGION}" = "PRC" ]; \
        then \
            pip config unset global.index-url; \
        fi

# Deploy TPCx-AI toolkit
RUN wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/scripts/bootstrap-tpcx-ai.sh -O /tmp/bootstrap-tpcx-ai.sh \
    && bash /tmp/bootstrap-tpcx-ai.sh --head \
    && rm -rf /tmp/bootstrap-tpcx-ai.sh

RUN if [ "${CLOUDTIK_REGION}" = "PRC" ]; \
        then \
            pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/; \
        fi
