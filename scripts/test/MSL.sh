root_path=./dataset
model_id=${1:-0}

python -u run.py \
--mode test \
--configs_path ./configs/ \
--save_path ./test_results/ \
--root_path $root_path \
--data MSL \
--data_origin DADA \
--gpu 0 \
--id $model_id