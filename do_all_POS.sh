collection=msrvtt10k
space=concept
visual_feature=resnext101-resnet152
rootpath=/home/mrim/deviv/irim/pytorch/danieljf24/VisualSearch
latent_space_size=1536
concept_space_size=512
overwrite=0
exp_name=concept_POS_WN
run=$1
postfix=runs_$run
postagger=WN
gpu=0

export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --text_mapping_layers 0-$latent_space_size \
                                            --visual_mapping_layers 0-$latent_space_size --tag_vocab_size $concept_space_size \
                                            --collection $collection --visual_feature $visual_feature --space $space \
                                            --exp_name $exp_name --postfix $postfix \
                                            --use_postag_vocab $postagger

