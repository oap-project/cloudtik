ARG BASE_IMAGE="nightly"
FROM cloudtik/cloudtik-deps:"$BASE_IMAGE"

ENV RUNTIME_PATH /home/cloudtik/runtime
RUN mkdir -p $RUNTIME_PATH
WORKDIR /home/cloudtik/runtime

ENV MYSQL_MAJOR 8.0
ENV MYSQL_VERSION 8.0.34-1debian11

RUN set -eux; \
# gpg: key 3A79BD29: public key "MySQL Release Engineering <mysql-build@oss.oracle.com>" imported
	key='859BE8D7C586F538430B19C2467B942D3A79BD29'; \
	export GNUPGHOME="$(mktemp -d)"; \
	sudo gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
	sudo mkdir -p /etc/apt/keyrings; \
	sudo gpg --batch --export "$key" | sudo tee /etc/apt/keyrings/mysql.gpg >/dev/null; \
	sudo gpgconf --kill all; \
	rm -rf "$GNUPGHOME"

RUN echo 'deb [ signed-by=/etc/apt/keyrings/mysql.gpg ] http://repo.mysql.com/apt/debian/ bullseye mysql-8.0' \
    | sudo tee /etc/apt/sources.list.d/mysql.list >/dev/null

# the "/var/lib/mysql" stuff here is because the mysql-server postinst doesn't have an explicit way to disable the mysql_install_db codepath besides having a database already "configured" (ie, stuff in /var/lib/mysql/mysql)
# also, we set debconf keys to make APT a little quieter
RUN { \
		echo mysql-community-server mysql-community-server/data-dir select ''; \
		echo mysql-community-server mysql-community-server/root-pass password ''; \
		echo mysql-community-server mysql-community-server/re-root-pass password ''; \
		echo mysql-community-server mysql-community-server/remove-test-db select false; \
	} | sudo debconf-set-selections \
	&& sudo apt-get update \
	&& sudo apt-get install -y \
		mysql-community-client="${MYSQL_VERSION}" \
		mysql-community-server-core="${MYSQL_VERSION}" \
	&& sudo rm -rf /var/lib/apt/lists/*
