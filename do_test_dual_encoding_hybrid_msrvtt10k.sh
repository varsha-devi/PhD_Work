rootpath=/home/mrim/deviv/irim/pytorch/danieljf24/VisualSearch
collectionStrt=single
testCollection=msrvtt10k
logger_name=/home/mrim/deviv/irim/pytorch/danieljf24/VisualSearch/msrvtt10k/train_results/hybrid_POS_treetagger/runs_6
overwrite=0

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

