# Stock Report Docker Setup

## Prerequisites


## Build the Docker image

```bash
docker build -t stock-report .
```

## Run the container

### Option 1: Connect to host's PostgreSQL (macOS/Linux)

```bash
docker run -p 9501:9501 \
  --add-host=host.docker.internal:host-gateway \
  stock-report
```

Then update the connection string in stock_report.py to use `host.docker.internal` instead of `localhost`.

### Option 2: Connect to external PostgreSQL

```bash
docker run -p 9501:9501 \
  -e DB_HOST=your-postgres-host \
  -e DB_PORT=5432 \
  -e DB_USER=$DB_USER \
  -e DB_PASSWORD=$DB_PASSWORD \
  -e DB_NAME=$DB_NAME \
  stock-report
```

### Option 3: Use Docker network (if PostgreSQL is also in Docker)

```bash
docker run -p 9501:9501 \
  --network your-postgres-network \
  stock-report
```

## Access the application

Open your browser and navigate to: http://localhost:9501

## Stop the container

```bash
docker ps  # Find the container ID
docker stop <container-id>
```

## Notes

- The application connects to PostgreSQL at `localhost:5432` by default
- Update the `CONNECTION_URI` in stock_report.py to match your PostgreSQL setup
- Port 9501 is exposed for Streamlit web interface
