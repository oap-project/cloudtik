# The development Docker image build a self-contained Cloudtik instance suitable
# for developers that need the source code to actively modify.

FROM cloudtik/cloudtik-deps:nightly
ADD cloudtik.tar /cloudtik
COPY git-rev /cloudtik/git-rev

# Install dependencies needed to build
RUN sudo apt-get update && sudo apt-get install -y curl unzip cmake gcc g++ && sudo apt-get clean
RUN sudo chown -R cloudtik:users /cloudtik && cd /cloudtik && git init

WORKDIR /cloudtik/
RUN bash build.sh
WORKDIR /cloudtik/python/
RUN export PATH="$HOME/anaconda3/envs/$CLOUDTIK_ENV/bin:$PATH" \
    && pip install -e .
WORKDIR /cloudtik
