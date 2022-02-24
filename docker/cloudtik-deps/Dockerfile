ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik-base:"$BASE_IMAGE"
ARG WHEEL_PATH
# For Click
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
COPY $WHEEL_PATH .
RUN $HOME/anaconda3/bin/pip --no-cache-dir install \
    $(basename $WHEEL_PATH)[all] \
    $(echo \
        "six==1.16.0" \
        "boto3==1.4.8" \
        "google-api-python-client==1.7.8" \
        "google-oauth" \
        "kubernetes" \
        "azure-cli-core==2.33.1" \
        "azure-identity==1.7.1" \
        "azure-mgmt-compute==23.1.0" \
        "azure-mgmt-network==19.3.0" \
        "azure-mgmt-resource==20.0.0" \
        "msrestazure==0.6.4") \
    $(if [ $($HOME/anaconda3/bin/python -c "import sys; print(sys.version_info.minor)") != 6 ]; then echo "kopf"; fi) \
    && $HOME/anaconda3/bin/pip uninstall cloudtik -y && sudo rm $(basename $WHEEL_PATH)
