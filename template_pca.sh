export PATH=/home/mrim/deviv/ws_dual_py3/bin:$PATH
collectionStrt=@@@collectionStrt@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
overwrite=@@@overwrite@@@

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python pca.py --collectionStrt $collectionStrt --testCollection $testCollection --overwrite $overwrite --logger_name $logger_name --save_pca