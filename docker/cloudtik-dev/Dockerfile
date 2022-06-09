# The development Docker image build a self-contained Cloudtik instance suitable
# for developers that need the source code to actively modify.

FROM cloudtik/cloudtik-deps:nightly

COPY cloudtik.tar /tmp/cloudtik.tar

# Install dependencies needed to build
RUN sudo apt-get update && sudo apt-get install -y curl unzip cmake gcc g++ && sudo apt-get clean

WORKDIR /cloudtik/
COPY git-rev ./git-rev
RUN tar -xf /tmp/cloudtik.tar && sudo chown -R cloudtik:users /cloudtik && git init \
    && bash build.sh
WORKDIR /cloudtik/python/
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && pip install -e .
WORKDIR /cloudtik
