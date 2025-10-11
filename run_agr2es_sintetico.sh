# Dataset V2 - Sin balance
echo "1. NLLB-600M | Dataset V2 | Sin balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v3 --model_name facebook/nllb-200-distilled-600M --epochs 20 --batch_size 4 --learning_rate 3e-5 --patience 3 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)
#CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v3 --model_name facebook/nllb-200-distilled-600M --epochs 12 --batch_size 8 --learning_rate 3e-5 --patience 3 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)
