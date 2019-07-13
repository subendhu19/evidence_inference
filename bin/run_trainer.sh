#!/usr/bin/env bash

python src/scripts/elmo_trainer.py --emb_size 8 --batch_size 8 --model elmo
python src/scripts/bert_trainer.py --batch_size 8 --model bert
