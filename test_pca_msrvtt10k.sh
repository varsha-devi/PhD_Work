export PATH=/home/mrim/deviv/ws_dual_py3/bin:$PATH
collectionStrt=single
testCollection=msrvtt10k
logger_name=/home/mrim/tools/irim/pytorch/dual_encoding_experiments/VisualSearchResults/msrvtt10k/train_results/latent_2048_pca/runs_0/
overwrite=0

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python pca.py --collectionStrt $collectionStrt --testCollection $testCollection --overwrite $overwrite --logger_name $logger_name --save_pca
