rootpath=/home/mrim/deviv/irim/pytorch/danieljf24/VisualSearch
collectionStrt=single
testCollection=msrvtt10k
logger_name=/home/mrim/deviv/irim/pytorch/danieljf24/VisualSearch/msrvtt10k/train_results/concept_256_xi/runs_5
overwrite=0
save_emb=concept_only_256.pth
gpu=0
export PATH=/home/mrim/quenot/anaconda3/envs/ws_dual_py3/bin:$PATH

CUDA_VISIBLE_DEVICES=$gpu python tester.py --collectionStrt $collectionStrt --testCollection $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --save_embs $save_emb

