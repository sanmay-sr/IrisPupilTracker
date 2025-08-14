#!/bin/bash

# Create Streamlit configuration directory
mkdir -p ~/.streamlit/

# Create Streamlit configuration file
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "port = $PORT" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

# Set file permissions
chmod 644 ~/.streamlit/config.toml

echo "Streamlit configuration completed successfully!"
