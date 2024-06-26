collection=$1
space=$2
visual_feature=$3
rootpath=$4
overwrite=1
num_epochs=1
exp_name=$5
postfix=$6
tag_vocab_size=$7
postagger=$8
# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature --space $space \
                                            --num_epochs $num_epochs --exp_name $exp_name --postfix $postfix --tag_vocab_size $tag_vocab_size \
                                            --use_postag_vocab $postagger

