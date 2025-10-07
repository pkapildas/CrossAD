root_path=./dataset/UCR_Anomaly_FullData

python -u run_UCR.py \
--mode train \
--configs_path ./configs/ \
--save_path ./test_results/ \
--root_path $root_path \
--data_origin UCR \
--gpu 0
