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

# Run the first Python file in the background with nohup
echo "Running $PYTHON_FILE_1 in the background..."
nohup python3 $PYTHON_FILE_1 > "$LOG_FILE_1" 2>&1 &
PID1=$!

# Wait for the first script to finish
wait $PID1

# Check if the first script completed successfully
if [ $? -ne 0 ]; then
  echo "Error: $PYTHON_FILE_1 failed to execute. Aborting."
  exit 1
fi

echo "$PYTHON_FILE_1 completed successfully."

# Run the second Python file in the background with nohup
echo "Running $PYTHON_FILE_2 in the background..."
nohup python3 $PYTHON_FILE_2 > "$LOG_FILE_2" 2>&1 &
PID2=$!

echo "$PYTHON_FILE_2 is running in the background with PID $PID2."
echo "Logs are being written to $LOG_FILE_1 and $LOG_FILE_2."
