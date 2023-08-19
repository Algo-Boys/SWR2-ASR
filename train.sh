#!/bin/sh

yes no | python -m swr2_asr.train --batch_size=8 --world_size=2 --dataset_path=/mnt/lustre/mladm/mfa252/data
