
function create_database_schema() {
    DATABASE_NAME=mlflow
    if [ "${SQL_DATABASE_ENGINE}" == "mysql" ]; then
        mysql --host=${SQL_DATABASE_HOST} --port=${SQL_DATABASE_PORT} --user=${SQL_DATABASE_USERNAME} --password=${SQL_DATABASE_PASSWORD}  -e "
                CREATE DATABASE IF NOT EXISTS ${DATABASE_NAME};" > ${MLFLOW_HOME}/logs/configure.log
    else
        # Use psql to create the database
        echo "SELECT 'CREATE DATABASE ${DATABASE_NAME}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DATABASE_NAME}')\gexec" | PGPASSWORD=${SQL_DATABASE_PASSWORD} \
            psql \
            --host=${SQL_DATABASE_HOST} \
            --port=${SQL_DATABASE_PORT} \
            --username=${SQL_DATABASE_USERNAME} > ${MLFLOW_HOME}/logs/configure.log
    fi
    # Future improvement: mlflow db upgrade [db_uri]
}

function init_schema() {
    if [ "${SQL_DATABASE}" == "true" ] \
      && [ "$AI_WITH_SQL_DATABASE" != "false" ]; then
        create_database_schema
    fi
}
