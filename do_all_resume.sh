collection=$1
space=$2
visual_feature=$3
rootpath=$4
overwrite=1
num_epochs=6
resume=$5

# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer_v1.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature --space $space \
                                            --num_epochs $num_epochs --resume $resume
