ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik-deps:"$BASE_IMAGE"

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

ENV PG_MAJOR 15
ENV PG_VERSION 15.4-1.pgdg110+1

# make the "en_US.UTF-8" locale so postgres will be utf-8 enabled by default
RUN set -eux; \
	sudo apt-get update; \
	sudo apt-get install -y --no-install-recommends locales; \
	sudo rm -rf /var/lib/apt/lists/*; \
	sudo localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

RUN set -eux; \
	sudo apt-get update; \
	sudo apt-get install -y --no-install-recommends \
		libnss-wrapper \
		zstd \
	; \
	sudo rm -rf /var/lib/apt/lists/*

RUN set -ex; \
# pub   4096R/ACCC4CF8 2011-10-13 [expires: 2019-07-02]
#       Key fingerprint = B97B 0AFC AA1A 47F0 44F2  44A0 7FCC 7D46 ACCC 4CF8
# uid                  PostgreSQL Debian Repository
	key='B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8'; \
	export GNUPGHOME="$(mktemp -d)"; \
	sudo gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
	sudo mkdir -p /etc/apt/keyrings; \
	sudo gpg --batch --export --armor "$key" | sudo tee /etc/apt/keyrings/mysql.gpg >/dev/null; \
	sudo gpgconf --kill all; \
	rm -rf "$GNUPGHOME"

RUN echo "deb [ signed-by=/usr/local/share/keyrings/postgres.gpg.asc ] http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" \
    | sudo tee /etc/apt/sources.list.d/postgres.list >/dev/null

sudo apt-get -qq update -y \
    && sudo apt-get install -y \
      --no-install-recommends postgresql-common \
    && sudo sed -ri 's/#(create_main_cluster) .*$/\1 = false/' /etc/postgresql-common/createcluster.conf \
    && sudo apt-get install -y \
        --no-install-recommends "postgresql-$PG_MAJOR" "postgresql-client-$PG_MAJOR" \
    && sudo rm -f /etc/apt/sources.list.d/postgres.list

ENV PATH $PATH:/usr/lib/postgresql/$PG_MAJOR/bin
