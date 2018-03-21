#!/usr/bin/env bash
python code/main_ensemble.py \
 --json_in_path=data/dev-v1.1.json \
 --ensemble_dir=experiments \
 --ensemble_folder_names=fulldp,fulldp1,fulldp2,fulldp3,fulldp4,fulldp5,fulldp6,fulldp7,fulldp8,fulldp9 \
 --ensemble_class_names=RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel,RNetPtrModel \
 --hidden_size_list=200,200,200,200,200,200,200,200,200,200 \
 --context_len_list=450,450,450,450,450,450,450,450,450,450 \
 --embedding_size_list=300,300,300,300,300,300,300,300,300,300

python code/evaluate.py data/dev-v1.1.json predictions.json

ensemble_folder_names=(fulldp fulldp1 fulldp2 fulldp3 fulldp4 fulldp5 fulldp6 fulldp7 fulldp8 fulldp9)

for name in "${ensemble_folder_names[@]}"
do
    python code/evaluate.py data/dev-v1.1.json predictions_${name}.json
#    echo fulldp-${name}-predictions.json
#    python code/evaluate.py data/dev-v1.1.json fulldp-${name}-predictions.json
done