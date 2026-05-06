#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs_stage2"
SAVE_DIR="pths_stage2"
mkdir -p "$LOG_DIR" "$SAVE_DIR"

NOHUP_LOG="$LOG_DIR/nohup_train.log"
PID_FILE="$LOG_DIR/train.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Stage2 training already running (PID $(cat "$PID_FILE")). Abort."
    exit 1
fi

nohup python -u train_distill.py \
    --data_root ./ \
    --lmdb_name DocTamperV1-TrainingSet \
    --minq 75 \
    --teacher_pth pths/dtd_doctamper.pth \
    --resume pths/best.pth \
    --reset_optimizer \
    --epochs 30 \
    --batch_size 48 \
    --eval_batch_size 8 \
    --num_workers 8 \
    --lr 5e-5 \
    --eta_min 1e-6 \
    --weight_decay 1e-4 \
    --alpha 1.0 \
    --beta 0.5 \
    --gamma 0.1 \
    --save_dir "$SAVE_DIR" \
    --save_interval 5 \
    --log_dir "$LOG_DIR" \
    > "$NOHUP_LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "Stage2 training started (PID $(cat "$PID_FILE")), log: $NOHUP_LOG"
echo "Monitor: tail -f $NOHUP_LOG"
echo "Stop:    kill \$(cat $PID_FILE)"
