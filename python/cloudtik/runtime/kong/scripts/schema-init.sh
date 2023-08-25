
function create_database_schema() {
    DATABASE_NAME=kong
    DATABASE_USER=kong
    # TODO: allow user to specify the database password
    DATABASE_PASSWORD=kong
    # Use psql to create the user and database
    DATABASE_EXISTS=$(PGPASSWORD=${SQL_DATABASE_PASSWORD} psql -lqt --host=${SQL_DATABASE_HOST} \
        --port=${SQL_DATABASE_PORT} \
        --username=${SQL_DATABASE_USERNAME} | cut -d \| -f 1 | grep -w $DATABASE_NAME | wc -l) || true
    if [[ $DATABASE_EXISTS == 0 ]]; then
        echo "CREATE USER $DATABASE_USER WITH PASSWORD '$DATABASE_PASSWORD'\gexec" | PGPASSWORD=${SQL_DATABASE_PASSWORD} \
            psql \
            --host=${SQL_DATABASE_HOST} \
            --port=${SQL_DATABASE_PORT} \
            --username=${SQL_DATABASE_USERNAME} > ${KONG_HOME}/logs/configure.log
        echo "SELECT 'CREATE DATABASE ${DATABASE_NAME} OWNER $DATABASE_USER' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DATABASE_NAME}')\gexec" | PGPASSWORD=${SQL_DATABASE_PASSWORD} \
            psql \
            --host=${SQL_DATABASE_HOST} \
            --port=${SQL_DATABASE_PORT} \
            --username=${SQL_DATABASE_USERNAME} > ${KONG_HOME}/logs/configure.log
    fi
}

function init_schema() {
    create_database_schema

    ADMIN_PASSWORD=kong
    KONG_PASSWORD=$ADMIN_PASSWORD sudo -E env "PATH=$PATH" \
      kong migrations bootstrap -c ${KONG_CONFIG_FILE} > ${KONG_HOME}/logs/configure.log  2>&1
}
