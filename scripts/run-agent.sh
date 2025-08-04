#!/bin/bash

# Navigate to the backend directory
cd "$(dirname "$0")/../backend" || exit 1

# Activate the virtual environment
source .venv/bin/activate

# Run the backend server
.venv/bin/python server.py
