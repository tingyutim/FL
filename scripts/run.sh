#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run server
python algos/$1/server.py &

# Run clients
for i in $(seq 1 $2); do
    python algos/$1/client.py &
done

wait
