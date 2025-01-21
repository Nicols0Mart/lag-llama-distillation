#!/bin/bash

# Process ID to monitor
# PID_TO_WATCH=10363

PID_TO_WATCH=1305893


# Command to run after the process terminates
COMMAND_TO_RUN="python test_pretrain_convergence.py --fold 0.70"
echo "current python is `(which python)`"
# Function to check if the process is running
is_process_running() {
    ps -p $1 > /dev/null 2>&1
    return $?
}



# Loop until the process terminates
while is_process_running $PID_TO_WATCH
do
    echo "Waiting for process $PID_TO_WATCH to terminate..."
    sleep 300
done

echo "Process $PID_TO_WATCH has terminated."
echo "Running the specified command:"
eval $COMMAND_TO_RUN
