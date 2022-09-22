#!/bin/bash

# Helper script to use tmux command to check session existence

session_name=$1
shift

tmux has-session -t $session_name >/dev/null 2>&1

if [ $? != 0 ]; then
  # session not exists
  echo "$session_name not found."
else
  echo "$session_name found."
fi

exit 0
