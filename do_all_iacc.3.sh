IRIM=/home/mrim/tools/irim
EXPE_ROOT=$IRIM/pytorch/dual_encoding_experiments
CODE_ROOT=$EXPE_ROOT/hybrid_space
RESULTS_ROOT=$EXPE_ROOT/VisualSearchResults
rootpath=$RESULTS_ROOT

collectionStrt=multiple
trainCollection=tgif-msrvtt10k
testCollection=iacc.3

run=$2
overwrite=0

exp_name=$1
postfix=runs_$run

train_results_path=$trainCollection/train_results/$exp_name
test_results_path=$testCollection/test_results/$exp_name
logger_name=$RESULTS_ROOT/$train_results_path/$postfix

export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

# testing
cd $CODE_ROOT
echo "Testing."
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

time python tester_avs.py --logger_name $logger_name --rootpath $rootpath \
--collectionStrt $collectionStrt --testCollection $testCollection \
--log_file logging_save --overwrite $overwrite \
--query_sets tv16.avs.txt,tv17.avs.txt,tv18.avs.txt

echo "Done."
