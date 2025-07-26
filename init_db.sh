#!/bin/bash
set -e

echo "Starting database initialization..."

# Function to check if database exists
database_exists() {
    local db_name=$1
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" -tAc "SELECT 1 FROM pg_database WHERE datname='$db_name'" | grep -q 1
}

# Create mlflow database if it doesn't exist
if ! database_exists "mlflow"; then
    echo "Creating mlflow database..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" -c "CREATE DATABASE mlflow;"
    echo "mlflow database created successfully"
else
    echo "mlflow database already exists"
fi

# Create prefect database if it doesn't exist
if ! database_exists "prefect"; then
    echo "Creating prefect database..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" -c "CREATE DATABASE prefect;"
    echo "prefect database created successfully"
else
    echo "prefect database already exists"
fi

echo "Database initialization completed successfully"
