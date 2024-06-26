IRIM=/home/mrim/tools/irim
EXPE_ROOT=$IRIM/pytorch/dual_encoding_experiments
CODE_ROOT=$EXPE_ROOT/hybrid_space
DATA_ROOT=$EXPE_ROOT/VisualSearchData
RESULTS_ROOT=$EXPE_ROOT/VisualSearchResults
TMP_ROOT=/dev/shm/dual_encoding_experiments/VisualSearch

collectionStrt=multiple
trainCollection=tgif-msrvtt10k
valCollection=tv2016train
testCollection=iacc.3
visual_feature=resnext101-resnet152

rootpath=$TMP_ROOT

latent_space_size=1536
concept_space_size=512

overwrite=0
model=dual_encoding
space=hybrid

exp_name=hybrid_${trainCollection}_${latent_space_size}_${concept_space_size}_xin
postfix=runs_$1

train_results_path=$trainCollection/train_results/$exp_name
test_results_path=$testCollection/test_results/$exp_name

export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

echo "Moving training data."
mkdir -p $TMP_ROOT/trc
time rsync -a $DATA_ROOT/$trainCollection $TMP_ROOT
time rsync -a $DATA_ROOT/$valCollection $TMP_ROOT
time rsync -a $DATA_ROOT/word2vec $TMP_ROOT
echo "Linking testing data."
if [ ! -e "$TMP_ROOT/$testCollection" ]; then
  ln -s $DATA_ROOT/$testCollection $TMP_ROOT
fi

# training
cd $CODE_ROOT
echo "Training / testing."
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

time python trainer.py --collectionStrt $collectionStrt --trainCollection $trainCollection \
--valCollection $valCollection --testCollection $testCollection --max_violation \
--text_norm --visual_norm --rootpath $rootpath --overwrite $overwrite \
--visual_feature $visual_feature --model $model --space $space --exp_name $exp_name \
--postfix $postfix --tag_vocab_size $concept_space_size \
--text_mapping_layers 0-$latent_space_size --visual_mapping_layers 0-$latent_space_size \
--concept_xi --classification_loss_weight 0.0  \
>& $TMP_ROOT/trc/${exp_name}_$postfix
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
sleep 100000
