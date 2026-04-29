#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" pths

NOHUP_LOG="$LOG_DIR/nohup_train.log"
PID_FILE="$LOG_DIR/train.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Training already running (PID $(cat "$PID_FILE")). Abort."
    exit 1
fi

nohup python -u train_distill.py \
    --data_root ./ \
    --lmdb_name DocTamperV1-TrainingSet \
    --minq 75 \
    --teacher_pth pths/dtd_doctamper.pth \
    --epochs 50 \
    --batch_size 48 \
    --eval_batch_size 8 \
    --num_workers 8 \
    --lr 1e-4 \
    --save_dir pths \
    --save_interval 5 \
    --log_dir "$LOG_DIR" \
    > "$NOHUP_LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "Training started (PID $(cat "$PID_FILE")), log: $NOHUP_LOG"
echo "Monitor: tail -f $NOHUP_LOG"
echo "Stop:    kill \$(cat $PID_FILE)"
