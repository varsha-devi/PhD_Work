import os
import sys
import json
import time
import torch
import pickle
import logging
import argparse
import shutil
import numpy as np

import evaluation
from model import get_model
from validate import norm_score, cal_perf, get_mod_embeddings, get_metrics

import util.data_provider as data
from util.vocab import Vocabulary
import util.metrics as metrics
from util.text2vec import get_text_encoder


from basic.util import read_dict, log_config
from basic.constant import ROOT_PATH
from basic.bigfile import BigFile
from basic.common import makedirsforfile, checkToSkip
from basic.generic_utils import Progbar


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='multiple', help='collection structure (single|multiple)')
    parser.add_argument('--split', default='test', type=str, help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth', type=str, help='name of checkpoint (default: model_best.pth)')
    parser.add_argument('--query_sets', type=str, default='tv16.avs.txt',  help='test query sets,  tv16.avs.txt,tv17.avs.txt,tv18.avs.txt for TRECVID 16/17/18.')
    #added
    parser.add_argument('--save_pred', action='store_true', help='save the pred error matrix and txt files: default is false')
    parser.add_argument('--save_embs', default=None, type=str, help='save video and captions embeddings: default is None')
    parser.add_argument('--load_embs', default=None, type=str, help='load video and captions embeddings: default is None')
    parser.add_argument('--log_file', default='logging', type=str, help='logging file name')
    parser.add_argument('--dump_errors', action='store_true', help='save the test video and captions errors: default is false')
    parser.add_argument('--pca', action='store_true', help='evaluate PCA on the video and textual features: default is false')
    parser.add_argument('--evaluate_pca', action='store_true',
                        help='evaluate PCA on the video and textual features: default is false')
    parser.add_argument('--truncate_latent', default=0, type=int,
                        help='Size of latent space for evaluation: default is 0 (do not truncate).')
    parser.add_argument('--truncate_concept', default=0, type=int,
                         help='Size of concept space for evaluation: default is 0 (do not truncate).')
    parser.add_argument('--power', default=1.0, type=float, help='power factor probs = sigmoid(x)^power')
    parser.add_argument('--shift', default=0.0, type=float, help='shift factor probs = sigmoid(x+shift)')
    parser.add_argument('--scale', default=1.0, type=float, help='scale factor probs => sigmoid(Ax)')
    parser.add_argument('--topk', default=0, type=int, help='Number of concept contibutors to be used for jaccard computation')
    
    #added

    args = parser.parse_args()
    return args


def get_vid_loaders_avs(opt, options, checkpoint):
    rootpath = opt.rootpath
    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname['test'] = testCollection
    collectionStrt = opt.collectionStrt
    
    visual_feat_file = BigFile(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature))
    assert options.visual_feat_dim == visual_feat_file.ndims
    video2frame = read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.visual_feature, 'video2frames.txt'))
    vid_data_loader = data.get_vis_data_loader(visual_feat_file, opt.batch_size, opt.workers, video2frame)
    
    return vid_data_loader



def get_model_embeds(options, checkpoint):
    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    return model


def eval_avs(t2v_matrix, caption_ids, video_ids, pred_result_file, rootpath, testCollection, query_set):

    inds = np.argsort(t2v_matrix, axis=1)
    with open(pred_result_file, 'w') as fout:
       for index in range(inds.shape[0]):
           ind = inds[index][::-1]
           fout.write(caption_ids[index]+' '+' '.join([video_ids[i]+' %s'%t2v_matrix[index][i]
               for i in ind])+'\n')

    templete = ''.join(open( 'tv-avs-eval/TEMPLATE_do_eval.sh').readlines())
    striptStr = templete.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@testCollection@@@', testCollection)
    striptStr = striptStr.replace('@@@topic_set@@@', query_set.split('.')[0])
    striptStr = striptStr.replace('@@@overwrite@@@', str(1))
    striptStr = striptStr.replace('@@@score_file@@@', pred_result_file)

    runfile = 'do_eval_%s.sh' % testCollection
    open(os.path.join('tv-avs-eval', runfile), 'w').write(striptStr + '\n')
    os.system('cd tv-avs-eval; chmod +x %s; bash %s; cd -' % (runfile, runfile))


def main():
    opt = parse_args()
    logging.info(json.dumps(vars(opt), indent=2))
    vis_embs = None

    rootpath = opt.rootpath
    collectionStrt = opt.collectionStrt
    assert collectionStrt == "multiple"
    testCollection = opt.testCollection
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)

    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(resume, checkpoint['epoch'], checkpoint['best_rsum']))
    options = checkpoint['opt']

    trainCollection = options.trainCollection
    valCollection = options.valCollection
    output_dir = resume.replace(trainCollection, testCollection)
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
    elif 'train_results' in output_dir:
        output_dir = output_dir.replace('/train_results/', '/' + opt.split + '_results/')
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/%s/' % (options.cv_name, trainCollection))
    output_dir = output_dir.replace('.pth', '')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prediction error matrix
    if opt.save_pred == True:
        pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth')
        if checkToSkip(pred_error_matrix_file, opt.overwrite):
            sys.exit(0)
    else:
        pred_error_matrix_file = None
        
    # Log configuration
    log_config(output_dir, ca=opt.log_file)
    logging.info(json.dumps(vars(opt), indent=2))
    
    # model
    model = get_model_embeds(options, checkpoint)
    model.val_start()

    # video data loader
    vid_data_loader = get_vid_loaders_avs(opt, options, checkpoint)

    # encode videos & queries
    start = time.time()
    vid_embs, vid_ids = get_mod_embeddings(options.space, vid_data_loader, model.embed_vis)
    logging.info("encode video time: %.3f s" % (time.time()-start))

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    for query_set in opt.query_sets.strip().split(','):
        output_dir_tmp = output_dir.replace(valCollection, '%s/%s/%s' % (query_set, trainCollection, valCollection))
        output_dir_tmp = output_dir_tmp.replace('/%s/' % options.cv_name, '/results/')
        pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score_%s.txt'%query_set[0:4])
        logging.info(pred_result_file)
        if checkToSkip(pred_result_file, opt.overwrite):
            sys.exit(0)
        makedirsforfile(pred_result_file)
        
        # query data loader
        query_file = os.path.join(rootpath, testCollection, 'TextData', query_set)
        query_loader = data.get_txt_data_loader(query_file, rnn_vocab, bow2vec, opt.batch_size, opt.workers)

        # encode videos & queries
        start = time.time()
        txt_embs, txt_ids = get_mod_embeddings(options.space, query_loader, model.embed_txt)
        logging.info("encode query time: %.3f s" % (time.time()-start))
            
        if options.space in ['hybrid', 'concept']:
            t2v_matrix_0 = evaluation.cal_simi(txt_embs[0], vid_embs[0])
            pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score_latent_%s.txt'%query_set[0:4])
            eval_avs(t2v_matrix_0, txt_ids, vid_ids, pred_result_file, rootpath, testCollection, query_set)

            t2v_matrix_1 = evaluation.cal_simi(txt_embs[1], vid_embs[1])
            pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score_concept_%s.txt'%query_set[0:4])
            eval_avs(t2v_matrix_1, txt_ids, vid_ids, pred_result_file, rootpath, testCollection, query_set)

            t2v_matrix_0 = norm_score(t2v_matrix_0)
            t2v_matrix_1 = norm_score(t2v_matrix_1)
            for w in [0.8]:
                print("\n")
                t2v_matrix = w * t2v_matrix_0 + (1-w) * t2v_matrix_1
                pred_result_file = os.path.join(output_dir_tmp, 'id.sent.score_%s_%.1f.txt' %(query_set[0:4], w))
                eval_avs(t2v_matrix, txt_ids, vid_ids, pred_result_file, rootpath, testCollection, query_set)
        else:
            t2v_matrix_0 = evaluation.cal_simi(txt_embs[0], vid_embs[0])
            eval_avs(t2v_matrix_0, txt_ids, vid_ids, pred_result_file, rootpath, testCollection, query_set)


if __name__ == '__main__':
    main()