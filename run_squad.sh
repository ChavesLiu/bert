#!/usr/bin/env bash
if [ ! -d squad_data ]; then
  mkdir -p squad_data
  wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
else:
  echo "squad_data has been downloaded !!"
fi

if [ ! -d chinese_L-12_H-768_A-12 ]; then
  wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
  unzip chinese_L-12_H-768_A-12.zip
else:
  echo "chinese bert model has been downloaded !!"
fi

bert_model_path=./chinese_L-12_H-768_A-12
python3 run_squad.py --vocab_file ${bert_model_path}/vocab.txt \
  --bert_config_file ${bert_model_path}/bert_config.json \
  --output_dir ./output \
  --do_train \
  --train_file squad_data/train-v1.1.json \
  --train_batch_size 16
