# Dataset V2 - Sin balance
echo "1. NLLB-600M | Dataset V2 | Sin balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-600M --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

# Dataset V1 - Con balance
echo "2. NLLB-600M | Dataset V1 | Con balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-600M --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3

# Dataset V2 - Sin balance
echo "3. NLLB-600M | Dataset V2 | Sin balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-1.3B --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

# Dataset V1 - Con balance
echo "4. NLLB-600M | Dataset V1 | Con balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-1.3B --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3

# Dataset V1 - Sin balance
echo "5.  NLLB-600M | Dataset V1 | Sin balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-600M --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

# Dataset V2 - Con balance
echo "6. NLLB-600M | Dataset V2 | Con balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-600M --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3

# Dataset V1 - Sin balance
echo "7. NLLB-600M | Dataset V1 | Sin balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v1 --model_name facebook/nllb-200-distilled-1.3B --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3 --config <(sed 's/balance_method: "weighted"/balance_method: "none"/' config.yaml)

# Dataset V2 - Con balance
echo "8. NLLB-600M | Dataset V2 | Con balance | AGR→ES"
CUDA_VISIBLE_DEVICES=1 python3 train.py --direction agr2es --dataset_version v2 --model_name facebook/nllb-200-distilled-1.3B --epochs 20 --batch_size 16 --learning_rate 3e-5 --patience 3
