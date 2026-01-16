#!/bin/bash

savepath=$1;

python main.py --mode=predict --dataset="usegeo" --seq_len=2 --db_seq_len=4 --batch_size=1 --arch_depth=6 --ckpt_dir="$savepath" --records_path=records
