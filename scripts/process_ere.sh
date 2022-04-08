export ERE_PATH="../../Dataset/ERE_EN/"
export OUTPUT_PATH="./processed_data/ere_bart"

mkdir $OUTPUT_PATH

python preprocessing/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH -s resource/splits/ERE-EN -b facebook/bart-large -w 1

export BASE_PATH="./processed_data/"
export SPLIT_PATH="./resource/low_resource_split/ere"

python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_001 -o $BASE_PATH/ere_bart/train.001.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_002 -o $BASE_PATH/ere_bart/train.002.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_003 -o $BASE_PATH/ere_bart/train.003.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $BASE_PATH/ere_bart/train.005.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $BASE_PATH/ere_bart/train.010.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $BASE_PATH/ere_bart/train.020.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $BASE_PATH/ere_bart/train.030.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $BASE_PATH/ere_bart/train.050.w1.oneie.json      
python preprocessing/split_dataset.py -i $BASE_PATH/ere_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_075 -o $BASE_PATH/ere_bart/train.075.w1.oneie.json
