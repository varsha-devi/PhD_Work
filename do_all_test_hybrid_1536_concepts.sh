IRIM=/home/mrim/tools/irim
EXPE_ROOT=$IRIM/pytorch/dual_encoding_experiments
CODE_ROOT=$EXPE_ROOT/hybrid_space
RESULTS_ROOT=$EXPE_ROOT/VisualSearchResults
rootpath=$RESULTS_ROOT

collectionStrt=single
testCollection=msrvtt10k

concept_space_size=$1
overwrite=0

exp_name=concept_dims_hybrid
postfix=runs_$concept_space_size

train_results_path=$testCollection/train_results/$exp_name
test_results_path=$testCollection/test_results/$exp_name
logger_name=$RESULTS_ROOT/$train_results_path/$postfix

export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

# testing
cd $CODE_ROOT
echo "Testing."
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

time python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

echo "Done."
