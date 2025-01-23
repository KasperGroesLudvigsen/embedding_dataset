#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <python_file_1> <python_file_2>"
  exit 1
fi

# Assign the command-line arguments to variables
PYTHON_FILE_1=$1
PYTHON_FILE_2=$2

# Run the first Python file with nohup
echo "Running $PYTHON_FILE_1 in the background..."
nohup python3 $PYTHON_FILE_1 > script1.log 2>&1 &

# Save the PID of the first script
PID1=$!

# Run the second Python file with nohup
echo "Running $PYTHON_FILE_2 in the background..."
nohup python3 $PYTHON_FILE_2 > script2.log 2>&1 &

# Save the PID of the second script
PID2=$!

echo "Scripts are running in the background."
echo "PID of $PYTHON_FILE_1: $PID1"
echo "PID of $PYTHON_FILE_2: $PID2"

echo "Logs are being written to script1.log and script2.log."
