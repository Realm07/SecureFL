#!/usr/bin/env bash
SESSION_NAME="fl-session"
SERVER_CMD="uvicorn src.server:app --reload"
CLIENT_CMD_BASE="python -m src.client --id"
NUM_CLIENTS=5

# helper to restart a command in an existing pane without respawn
restart_in_pane() {
  local pane="$1"   # e.g. fl-session:0.0 or a pane-id
  local cmd="$2"

  # Stop current process, force prompt onto a clean new line, aggressively clear line,
  # then send the command.
  tmux send-keys -t "$pane" C-c          # interrupt process
  sleep 0.08
  tmux send-keys -t "$pane" C-m          # press Enter to get a fresh prompt
  sleep 0.06
  tmux send-keys -t "$pane" C-a C-k      # go to start + kill to end (bash/zsh)
  sleep 0.04
  tmux send-keys -t "$pane" C-l          # redraw/clear screen (Ctrl-L)
  sleep 0.04
  tmux send-keys -t "$pane" "$cmd" C-m
  sleep 0.05
}

# server
restart_in_pane "${SESSION_NAME}:0.0" "$SERVER_CMD"

sleep 5

# clients (pane indexes assumed 0.1 .. 0.5)
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  pane="${SESSION_NAME}:0.$((i + 1))"
  restart_in_pane "$pane" "$CLIENT_CMD_BASE $i"
done
