ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ml-runtime:"$BASE_IMAGE"

ARG CLOUDTIK_REGION=""

RUN sudo sed -i 's/mirrors.aliyun.com/archive.ubuntu.com/' /etc/apt/sources.list \
    && pip config unset global.index-url

# Deploy TPCx-AI toolkit
RUN wget https://raw.githubusercontent.com/oap-project/cloudtik/main/tools/benchmarks/ml/tpcx-ai/scripts/bootstrap-tpcx-ai.sh -O /tmp/bootstrap-tpcx-ai.sh \
    && bash /tmp/bootstrap-tpcx-ai.sh --head \
    && rm -rf /tmp/bootstrap-tpcx-ai.sh

RUN if [ "${CLOUDTIK_REGION}" = "PRC" ]; \
        then \
            sudo sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
            && sudo sed -i 's/security.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
            && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/; \
        fi
