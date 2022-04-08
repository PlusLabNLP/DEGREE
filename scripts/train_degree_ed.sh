export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

DATASET="ace05e"
# DATASET="ace05ep"
# DATASET="ere"

python degree/generate_data_degree_ed.py -c config/config_degree_ed_${DATASET}.json
python degree/train_degree_ed.py -c config/config_degree_ed_${DATASET}.json
