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

# Install system dependencies for OpenCV and MediaPipe
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

echo "Streamlit configuration completed successfully!"
echo "System dependencies installed for OpenCV and MediaPipe compatibility"
