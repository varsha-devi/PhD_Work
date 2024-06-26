import os
import sys
import json
import torch
import pickle
import logging
import argparse
import shutil

import evaluation
from model import get_model
from validate import norm_score, cal_perf, get_mod_embeddings, get_metrics

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
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--testCollection', type=str, help='test collection')
    parser.add_argument('--collectionStrt', type=str, default='single', help='collection structure (single|multiple)')
    parser.add_argument('--split', default='test', type=str,
                        help='split, only for single-folder collection structure (val|test)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=100, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=5, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth', type=str,
                        help='name of checkpoint (default: model_best.pth)')
    parser.add_argument('--save_pred', action='store_true', help='save the pred error matrix and txt files: default is false')
    # parser.add_argument('--save_embs', default='emb.pth', type=str, help='save video and captions embeddings: default is None')
    parser.add_argument('--save_embs', default=None, type=str, help='save video and captions embeddings: default is None')
    parser.add_argument('--load_embs', default=None, type=str, help='load video and captions embeddings: default is None')
    parser.add_argument('--log_file', default='logging', type=str, help='logging file name')
    parser.add_argument('--dump_errors', action='store_false', help='save the test video and captions errors: default is false')
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

    args = parser.parse_args()
    return args


def get_data_loaders(opt, options, checkpoint):
    rootpath = opt.rootpath
    testCollection = opt.testCollection
    collections_pathname = options.collections_pathname
    collections_pathname[opt.split] = testCollection
    collectionStrt = opt.collectionStrt

    # data loader prepare
    test_cap = os.path.join(rootpath, collections_pathname[opt.split], 'TextData', '%s.caption.txt'%testCollection)
    if collectionStrt == 'single':
        test_cap = os.path.join(rootpath, collections_pathname[opt.split], 'TextData', '%s%s.caption.txt'%(testCollection,
                                                                                                        opt.split))
    elif collectionStrt == 'multiple':
        test_cap = os.path.join(rootpath, collections_pathname[opt.split], 'TextData', '%s.caption.txt'%testCollection)
    else:
        raise NotImplementedError('collection structure %s not implemented' % collectionStrt)

    caption_files = {opt.split: test_cap}
    img_feat_path = os.path.join(rootpath, collections_pathname[opt.split], 'FeatureData', options.visual_feature)
    visual_feats = {opt.split: BigFile(img_feat_path)}
    assert options.visual_feat_dim == visual_feats[opt.split].ndims
    video2frames = {opt.split: read_dict(os.path.join(img_feat_path, 'video2frames.txt'))}

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'vocabulary', 'bow',
                                  options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, collections_pathname['train'], 'TextData', 'vocabulary', 'rnn',
                                  options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)

    # set data loader
    vid_data_loader = data.get_vis_data_loader(visual_feats[opt.split], opt.batch_size, opt.workers, video2frames[opt.split],
                                               video_ids=data.read_video_ids(caption_files[opt.split]))
    text_data_loader = data.get_txt_data_loader(caption_files[opt.split], rnn_vocab, bow2vec, opt.batch_size, opt.workers)

    return vid_data_loader, text_data_loader


def get_model_embeds(options, checkpoint):
    # Construct the model
    model = get_model(options.model)(options)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']
    model.val_start()

    return model


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(resume, checkpoint['epoch'], checkpoint['best_rsum']))
    options = checkpoint['opt']
    if opt.topk == 0:
        opt.topk = options.tag_vocab_size
    
    # collection setting
    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    test_opt = opt
    
    if 'checkpoints' in output_dir:
        output_dir = output_dir.replace('/checkpoints/', '/results/')
    elif 'train_results' in output_dir:
        output_dir = output_dir.replace('/train_results/', '/%s_results/'%(str(opt.split)))
    else:
        output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/%s/' % (options.cv_name, trainCollection))
        
    output_dir = output_dir.replace('.pth', '')
    if opt.scale != 1.0 or opt.shift != 0.0 or opt.power != 1.0 or opt.topk != options.tag_vocab_size:
        output_dir = os.path.join(output_dir, str(opt.scale)+'_'+str(opt.shift)+'_'+str(opt.power), 'topk_'+str(opt.topk))
    
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

    # Get embeddings
    if opt.load_embs is None or opt.save_embs is not None:
        vid_data_loader, txt_data_loader = get_data_loaders(opt, options, checkpoint)
        model = get_model_embeds(options, checkpoint)
        model.val_start()
        vid_embs, vid_ids = get_mod_embeddings(vid_data_loader, model.embed_vis)
        txt_embs, txt_ids = get_mod_embeddings(txt_data_loader, model.embed_txt)

    # Save embeddings
    if opt.save_embs is not None:
        embs_file = os.path.join(output_dir, opt.save_embs)
        ids_file = os.path.join(output_dir, 'ids.pth')
        torch.save({'video_embs': vid_embs[0], 'video_tag_scores': vid_embs[1],
                    'cap_embs': txt_embs[0], 'cap_tag_scores': txt_embs[1],}, embs_file)
        torch.save({'video_ids': vid_ids, 'caption_ids': txt_ids,}, ids_file)
        logging.info("save embeddings into: %s" % embs_file)

    # Call to do something with the saved files goes here.
        
    # Load embeddings
    if opt.load_embs is not None:
        embs_file = os.path.join(output_dir, opt.load_embs)
        ids_file = os.path.join(output_dir, 'ids.pth')
        embs_dict = torch.load(embs_file)
        ids_dict = torch.load(ids_file)
        vid_embs = [embs_dict['video_embs'], embs_dict['video_tag_scores'], None]
        txt_embs = [embs_dict['cap_embs'], embs_dict['cap_tag_scores'], None]
        vid_ids, txt_ids = ids_dict['video_ids'], ids_dict['caption_ids']
        logging.info("load embeddings from: %s" % embs_file)

    # Save predicted tags
    if opt.save_pred == True and options.space in ['concept', 'hybrid']:
        vid_tag_probs = torch.sigmoid(torch.tensor(vid_embs[1])).numpy()
        txt_tag_probs = torch.sigmoid(torch.tensor(txt_embs[1])).numpy()
        if not hasattr(options, 'use_postag_vocab') or options.use_postag_vocab == None:
            tag_vocab_path = os.path.join(rootpath, options.collections_pathname['train'], 'TextData', 'tags', 'video_label_th_1')
        else:
            tag_vocab_path = os.path.join(rootpath, options.collections_pathname['train'], 'TextData',
                                          'tags_POS', 'tags_%s'%options.use_postag_vocab, 'video_label_th_1')
        
        logging.info(tag_vocab_path)
        evaluation.pred_tag(vid_tag_probs, vid_ids, tag_vocab_path, options.tag_vocab_size, os.path.join(output_dir, 'video'))
        evaluation.pred_tag(txt_tag_probs, txt_ids, tag_vocab_path, options.tag_vocab_size, os.path.join(output_dir, 'text'))
    
    # Truncate latent or concept vectors, makes sense after a PCA transformation.
    if opt.truncate_latent != 0:
        vid_embs[0], txt_embs[0] = vid_embs[0][:,:opt.truncate_latent], txt_embs[0][:,:opt.truncate_latent]
    if opt.truncate_concept!= 0:
        vid_embs[1], txt_embs[1] = vid_embs[1][:,:opt.truncate_concept], txt_embs[1][:,:opt.truncate_concept]

    embs, ids = [vid_embs, txt_embs], [vid_ids, txt_ids]
    get_metrics(options, embs, ids, pred_error_matrix_file=pred_error_matrix_file, test_opt=opt)
    

if __name__ == '__main__':
    main()
