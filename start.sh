#!/usr/bin/env bash
#
# start.sh – Run FL server + clients in tmux, or kill the session.

SESSION_NAME="fl-session"
VENV_ACTIVATE="source /home/shees/Downloads/Code/SecureFL/env_linux/bin/activate"
SERVER_CMD="uvicorn src.server:app --reload"
CLIENT_CMD_BASE="python -m src.client --id"
NUM_CLIENTS=5
SERVER_DELAY=1   # seconds

# --- Kill Option ---
if [ "$1" == "kill" ]; then
  echo "Killing tmux session: $SESSION_NAME"
  tmux kill-session -t $SESSION_NAME 2>/dev/null && echo "Session killed." || echo "No such session running."
  exit 0
fi

# --- Script Logic ---
tmux kill-session -t $SESSION_NAME 2>/dev/null

echo "Starting new tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME -n "FL"

# --- Server Pane ---
tmux send-keys -t $SESSION_NAME:0.0 "$VENV_ACTIVATE && $SERVER_CMD" C-m
tmux select-pane -t $SESSION_NAME:0.0 -T "Server"

# --- Delay before clients ---
echo "Waiting $SERVER_DELAY seconds for server to initialize..."
sleep $SERVER_DELAY

# --- Clients Pane(s) ---
right_pane=$(tmux split-window -h -t $SESSION_NAME:0.0 -P -F "#{pane_id}")
tmux send-keys -t "$right_pane" "$VENV_ACTIVATE && $CLIENT_CMD_BASE 0" C-m
tmux select-pane -t "$right_pane" -T "Client 0"

for i in $(seq 1 $(($NUM_CLIENTS - 1))); do
  new_pane=$(tmux split-window -v -t "$right_pane" -P -F "#{pane_id}")
  tmux send-keys -t "$new_pane" "$VENV_ACTIVATE && $CLIENT_CMD_BASE $i" C-m
  tmux select-pane -t "$new_pane" -T "Client $i"
done

# Layout: big left pane, stacked right panes
tmux select-layout -t $SESSION_NAME:0 main-vertical

echo "Attaching to session. Use 'Ctrl-b d' to detach."
tmux attach-session -t $SESSION_NAME
