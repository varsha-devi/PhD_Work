IRIM=/home/mrim/tools/irim
EXPE_ROOT=$IRIM/pytorch/dual_encoding_experiments
CODE_ROOT=$EXPE_ROOT/hybrid_space
# RESULTS_ROOT=$EXPE_ROOT/VisualSearchResults
# rootpath=$RESULTS_ROOT
rootpath=/home/mrim/deviv/Github_Repos_SoA/Hybrid_Space_Dual_Encoding/VisualSearch/

collectionStrt=single
testCollection=msrvtt10k

# latent_space_size=$1
# overwrite=0

# exp_name=latent_dims
# postfix=runs_$latent_space_size

# train_results_path=$testCollection/train_results/$exp_name
# test_results_path=$testCollection/test_results/$exp_name
# logger_name=$RESULTS_ROOT/$train_results_path/$postfix
logger_name=/home/mrim/deviv/Github_Repos_SoA/Hybrid_Space_Dual_Encoding/VisualSearch/checkpoints

checkpoint_name=concept_cosine_512_runs_13.pth

topk=$1
power=$2
scale=$3
shift=$4
split=$5

export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

# testing
cd $CODE_ROOT
echo "Testing."
gpu=0
export CUDA_VISIBLE_DEVICES=$gpu

# time python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name
time python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --logger_name $logger_name --checkpoint_name $checkpoint_name --split $split --topk $topk --power $power --scale $scale --shift $shift

echo "Done."
