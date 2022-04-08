export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

DATASET="ace05e"
# DATASET="ace05ep"
# DATASET="ere"

python degree/generate_data_GenEAE.py -c config/config_GenEAE_$DATASET.json
python degree/train_GenEAE.py -c config/config_GenEAE_$DATASET.json
