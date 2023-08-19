#!/bin/sh

yes no | python -m swr2_asr.train --epochs=100 --batch_size=30 --dataset_path=/mnt/lustre/mladm/mfa252/data
