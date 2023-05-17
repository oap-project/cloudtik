ARG BASE_IMAGE="nightly"
FROM cloudtik/spark-ai-runtime:"$BASE_IMAGE"

COPY requirements.txt /tmp/requirements.txt
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    # Install machine learning and basic libraries
    && pip --no-cache-dir install -r /tmp/requirements.txt \
    && sudo rm /tmp/requirements.txt
    # Install DGL
    && pip install --no-cache-dir --pre dgl -f https://data.dgl.ai/wheels/repo.html \
    && pip install --no-cache-dir --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
