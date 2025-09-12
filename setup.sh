#!/usr/bin/env bash
# setup.sh – create tmux session with env active in all panes

SESSION_NAME="fl-session"
VENV_ACTIVATE="source /home/shees/Downloads/Code/SecureFL/env_linux/bin/activate"
NUM_CLIENTS=5

# Create new session only if it doesn't exist
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Session $SESSION_NAME already exists. Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "Starting persistent tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME -n "FL"

# Left pane: server shell
tmux send-keys -t $SESSION_NAME:0.0 "$VENV_ACTIVATE" C-m
tmux select-pane -t $SESSION_NAME:0.0 -T "Server"

# Right side: split once vertically for clients
right_pane=$(tmux split-window -h -t $SESSION_NAME:0.0 -P -F "#{pane_id}")
tmux send-keys -t "$right_pane" "$VENV_ACTIVATE" C-m
tmux select-pane -t "$right_pane" -T "Client 0"

# Remaining client panes stacked below
for i in $(seq 1 $(($NUM_CLIENTS - 1)))
do
  new_pane=$(tmux split-window -v -t "$right_pane" -P -F "#{pane_id}")
  tmux send-keys -t "$new_pane" "$VENV_ACTIVATE" C-m
  tmux select-pane -t "$new_pane" -T "Client $i"
done

# Layout: big left pane, stacked right panes
tmux select-layout -t $SESSION_NAME:0 main-vertical

echo "Env activated in all panes. Attach with:"
echo "  tmux attach -t $SESSION_NAME"
