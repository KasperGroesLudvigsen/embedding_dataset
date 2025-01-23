#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <python_file_1> <python_file_2>"
  exit 1
fi

# Assign the command-line arguments to variables
PYTHON_FILE_1=$1
PYTHON_FILE_2=$2

# Extract the base names of the Python files (without the directory and extension)
LOG_FILE_1=$(basename "$PYTHON_FILE_1" .py).log
LOG_FILE_2=$(basename "$PYTHON_FILE_2" .py).log

# Run the first Python file with nohup
echo "Running $PYTHON_FILE_1 in the background..."
nohup python3 $PYTHON_FILE_1 > "$LOG_FILE_1" 2>&1 &

# Save the PID of the first script
PID1=$!

# Run the second Python file with nohup
echo "Running $PYTHON_FILE_2 in the background..."
nohup python3 $PYTHON_FILE_2 > "$LOG_FILE_2" 2>&1 &

# Save the PID of the second script
PID2=$!

echo "Scripts are running in the background."
echo "PID of $PYTHON_FILE_1: $PID1"
echo "PID of $PYTHON_FILE_2: $PID2"

echo "Logs are being written to $LOG_FILE_1 and $LOG_FILE_2."
