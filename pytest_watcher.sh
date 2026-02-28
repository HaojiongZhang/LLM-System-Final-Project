#!/bin/bash
LOG=~/workspace/pytest_bg.log
OUT=~/workspace/pytest_watch.log
PIDFILE=~/workspace/pytest_bg.pid

echo "watch started at $(date)" >> "$OUT"
while true; do
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p $PID > /dev/null; then
      echo "$(date): pytest running (PID $PID)" >> "$OUT"
      tail -n 80 "$LOG" >> "$OUT" 2>/dev/null || true
    else
      echo "$(date): pytest process $PID not found, collecting final log" >> "$OUT"
      tail -n 400 "$LOG" >> "$OUT" 2>/dev/null || true
      grep -E "[0-9]+ failed|[0-9]+ passed|ERROR|Interrupted" "$LOG" | tail -n 50 >> "$OUT" 2>/dev/null || true
      break
    fi
  else
    echo "$(date): No PID file; attempting to find pytest process" >> "$OUT"
    PID=$(pgrep -f "pytest -q" | head -n1 || true)
    if [ -n "$PID" ]; then
      echo "$(date): pytest running (PID $PID) without pidfile" >> "$OUT"
      tail -n 80 "$LOG" >> "$OUT" 2>/dev/null || true
    else
      echo "$(date): No pytest process found; collecting final log and exiting" >> "$OUT"
      tail -n 400 "$LOG" >> "$OUT" 2>/dev/null || true
      grep -E "[0-9]+ failed|[0-9]+ passed|ERROR|Interrupted" "$LOG" | tail -n 50 >> "$OUT" 2>/dev/null || true
      break
    fi
  fi
  sleep 30
done

echo "watch finished at $(date)" >> "$OUT"
