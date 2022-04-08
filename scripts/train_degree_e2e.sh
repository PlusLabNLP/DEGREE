export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

DATASET="ace05e"
# DATASET="ace05ep"
# DATASET="ere"

python degree/generate_data_GenE2E.py -c config/config_GenE2E_$DATASET.json
python degree/train_GenE2E.py -c config/config_GenE2E_$DATASET.json
