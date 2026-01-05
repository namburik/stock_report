Summary of the solution

    Purpose
        A Streamlit web app that reads stock feature data from a PostgreSQL database, provides interactive filtering and visualizations, and includes an optional AI assistant for natural-language searches / context.

    Main components
        stock_report.py
            Streamlit app entrypoint and UI.
            Uses Plotly for visualizations and pandas for data handling.
        util.py
            LangChain / LLM integration utilities.
            StreamHandler — callback handler that streams LLM tokens into a Streamlit container.
            create_agent_aws() — builds a LangChain agent backed by ChatOllama (model "llama3.2:3b") and attempts to attach an external tool via a MultiServerMCPClient (duckduckgo MCP server). If MCP initialization fails, falls back to creating an agent without external tools.
        Dockerfile to containerize the Streamlit app.
        

    Key implementation details / choices
        Caching: st.cache_data used to cache database reads (TTL configured) to reduce DB load and speed interactive use.
        Data cleaning: explicit rules to treat sentinel/invalid values as null (alpha == 10000, atr > 20).
        LLM integration: uses a local Ollama endpoint (configurable for Docker vs host) and supports an optional duckduckgo tool via a MultiServerMCPClient to fetch recent web info. Temperature is set to 0 for deterministic responses.
        Streaming LLM output: StreamHandler streams tokens back to the Streamlit UI for a live typing effect.
        Async handling: the app creates and reuses an asyncio loop stored in st.session_state so LLM and tool calls can run without blocking the UI.

    How to run
        Locally: set Streamlit secrets (connections.postgres.url) to point at your Postgres, then run the Streamlit app or use the provided Docker image.
        Docker: README shows docker build and docker run examples. Streamlit is exposed on port 9501. When connecting to host Postgres from macOS/Linux, the README suggests using host.docker.internal and adding --add-host or using a Docker network for a Postgres container.


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


<img width="1467" height="788" alt="image" src="https://github.com/user-attachments/assets/da6a55d4-437c-400d-ad89-c1fab1ebaf9a" />

<img width="1464" height="806" alt="image" src="https://github.com/user-attachments/assets/795813c9-6624-4e47-9f97-f578b74709a8" />

