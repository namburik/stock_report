# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PostgreSQL and curl
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx && \
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx

# Copy requirements file
COPY requirements.txt .

# Copy the MCP server code
COPY duckduckgo_mcp_server /app/duckduckgo_mcp_server


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY stock_report.py .
COPY util.py .

# Copy Streamlit secrets configuration
COPY .streamlit/secrets.toml /app/.streamlit/secrets.toml

# Expose Streamlit default port
EXPOSE 9501

# Set environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=9501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV RUNNING_IN_DOCKER=true

# Run the Streamlit app
CMD ["streamlit", "run", "stock_report.py", "--server.port=9501", "--server.address=0.0.0.0"]
