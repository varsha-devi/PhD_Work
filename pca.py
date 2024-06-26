import os
import sys
import json
import torch
import pickle
import logging
import argparse
import shutil

import util.data_provider as data
from util.text2vec import get_text_encoder
import util.metrics as metrics
from util.vocab import Vocabulary

from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip



def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--testCollection', type=str, default= 'msrvtt10k', help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--logger_name', default='runs', help='Path where trained models are saved till runs_XX')
    parser.add_argument('--checkpoint_name', default='model_best.pth', type=str, help='name of checkpoint (default: model_best.pth)')
    parser.add_argument('--save_pca', action='store_true', help='save the test video and captions features: default is false')

    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    
    # collection setting
    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname['test'] = testCollection

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
        output_dir = output_dir.replace('.pth', '')
    elif 'train_results' in output_dir:
        output_dir = output_dir.replace('/train_results/', '/test_results/')
        output_dir = output_dir.replace('.pth', '')
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/%s/' % (options.cv_name, trainCollection))
        output_dir = output_dir.replace('.pth', '')
    
    pca_files = [os.path.join(output_dir, 'vemb_pca.pkl'),\
                 os.path.join(output_dir, 'cemb_pca.pkl'), \
                 os.path.join(output_dir, 'vprobs_pca.pkl'),\
                 os.path.join(output_dir, 'cprobs_pca.pkl')]
    
    if options.space in ['hybrid', 'concept'] and all(list(map(os.path.isfile,pca_files))):
        print("All PCA result files exist")
        sys.exit(0)
    if options.space in ['latent'] and all(list(map(os.path.isfile,pca_files[0:1]))):
        print("All PCA result files exist")
        sys.exit(0)
    
    if options.space in ['hybrid', 'concept']:
        with open(os.path.join(output_dir,'video_embs.pkl'),'rb') as f,\
        open(os.path.join(output_dir,'cap_embs.pkl'),'rb') as f2, \
        open(os.path.join(output_dir,'video_probs.pkl'),'rb') as f3, \
        open(os.path.join(output_dir,'cap_probs.pkl'),'rb') as f4:
            vf = pickle.load(f)
            cf = pickle.load(f2)
            vp = pickle.load(f3)
            cp = pickle.load(f4)
            print(vf.shape, cf.shape, vp.shape, cp.shape)
        
    elif options.space in ['latent']:
        with open(os.path.join(output_dir,'video_embs.pkl'),'rb') as f, open(os.path.join(output_dir,'cap_embs.pkl'),'rb') as f2:
            vf = pickle.load(f)
            cf = pickle.load(f2)
            print(vf.shape, cf.shape)

    # TODO
    #First do pca operations with vf, cf, vp, cp arrays
    #{start PCA
    # end PCA}
    
    if opt.save_pca == True:
        if options.space in ['hybrid', 'concept']:
             with open(os.path.join(output_dir, 'vemb_pca.pkl'),'wb') as f,\
                open(os.path.join(output_dir, 'cembs_pca.pkl'), 'wb') as f2, \
                open(os.path.join(output_dir, 'vprobs_pca.pkl'), 'wb') as f3,\
                open(os.path.join(output_dir,'cprobs_pca.pkl'), 'wb') as f4:
                    pickle.dump('#variable_containing_pca_components', f)
                    pickle.dump('#variable_containing_pca_components', f2)
                    pickle.dump('#variable_containing_pca_components', f3)
                    pickle.dump('#variable_containing_pca_components', f4)
        elif options.space in ['latent']:
            with open(os.path.join(output_dir,'vembs_pca.pkl'),'wb') as f, open(os.path.join(output_dir,'cembs_pca.pkl'), 'wb') as f2:
                pickle.dump('variable_containing_pca_components', f)
                pickle.dump('variable_containing_pca_components', f2)

if __name__ == '__main__':
    main()
