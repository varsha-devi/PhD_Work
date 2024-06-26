IRIM=/home/mrim/tools/irim
EXPE_ROOT=$IRIM/pytorch/dual_encoding_experiments
CODE_ROOT=$EXPE_ROOT/hybrid_space
DATA_ROOT=$EXPE_ROOT/VisualSearchData
RESULTS_ROOT=$EXPE_ROOT/VisualSearchResults
TMP_ROOT=/dev/shm/dual_encoding_experiments/VisualSearch

collection=msrvtt10k
space=concept
visual_feature=resnext101-resnet152
rootpath=$TMP_ROOT

latent_space_size=0
concept_space_size=512
overwrite=0

exp_name=concept_512_jxnn
postfix=runs_$1

train_results_path=$collection/train_results/$exp_name
test_results_path=$collection/test_results/$exp_name

export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

echo "Moving training data."
mkdir -p $TMP_ROOT/trc
time rsync -a $DATA_ROOT/ $TMP_ROOT

# training
cd $CODE_ROOT
echo "Training / testing."
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

time python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation \
--text_norm --visual_norm --text_mapping_layers 0-$latent_space_size \
--visual_mapping_layers 0-$latent_space_size --tag_vocab_size $concept_space_size \
--collection $collection --visual_feature $visual_feature --space $space --concept_xi \
--exp_name $exp_name --postfix $postfix --classification_loss_weight 0.0 \
>& $TMP_ROOT/trc/$postfix
echo "Moving train results."
mkdir -p $RESULTS_ROOT/$train_results_path
rm -rf $RESULTS_ROOT/$train_results_path/$postfix
rm -f $TMP_ROOT/$train_results_path/$postfix/model_last.pth
rm -f $TMP_ROOT/$train_results_path/$postfix/events*
time mv $TMP_ROOT/$train_results_path/$postfix $RESULTS_ROOT/$train_results_path
echo "Moving test results."
mkdir -p $RESULTS_ROOT/$test_results_path/$postfix
rm -rf $RESULTS_ROOT/$test_results_path/$postfix
time mv $TMP_ROOT/$test_results_path/$postfix $RESULTS_ROOT/$test_results_path
echo "Done."
