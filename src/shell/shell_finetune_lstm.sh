cd ../train
python3 finetune_main.py --model_type lstm --cluster_id 1 --model_index 1
python3 finetune_main.py --model_type lstm --cluster_id 2 --model_index 1
python3 finetune_main.py --model_type lstm --cluster_id 3 --model_index 1
python3 finetune_main.py --model_type lstm --cluster_id 4 --model_index 1

python3 finetune_main.py --model_type lstm --cluster_id 1 --model_index 2
python3 finetune_main.py --model_type lstm --cluster_id 2 --model_index 2
python3 finetune_main.py --model_type lstm --cluster_id 3 --model_index 2
python3 finetune_main.py --model_type lstm --cluster_id 4 --model_index 2

python3 finetune_main.py --model_type lstm --cluster_id 1 --model_index 3
python3 finetune_main.py --model_type lstm --cluster_id 2 --model_index 3
python3 finetune_main.py --model_type lstm --cluster_id 3 --model_index 3
python3 finetune_main.py --model_type lstm --cluster_id 4 --model_index 3