#!/usr/bin/env bash
# stop.sh – stop server + clients without killing tmux session

SESSION_NAME="fl-session"
NUM_CLIENTS=5

# Stop server
tmux send-keys -t $SESSION_NAME:0.0 C-c

# Stop clients
for i in $(seq 0 $(($NUM_CLIENTS - 1)))
do
  tmux send-keys -t $SESSION_NAME:0.$(($i + 1)) C-c
done

echo "Stopped server + $NUM_CLIENTS clients. Env still active in all panes."
