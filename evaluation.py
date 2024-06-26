import os
import json
import torch
import numpy as np
from loss import jaccard_sim, jaccard_sim_transform
from scipy.spatial import distance
from basic.generic_utils import Progbar
from basic.common import makedirsforfile
import math

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 * X / norm


# If the number of videos or captions are too large, the memory may be not enough for jaccard similarity computation.
# Hence, we split the sentence embedding matrix into a sequence of matrices with smaller size
# Modified by GQ, replaces the cal_error_batch, better control of the used memory for large numbers of concepts

# def jaccard_sim_batch(im, s, max_mem=2147483648):
#     batch_count = math.ceil(im.shape[0]*s.shape[0]*im.shape[1]*4/max_mem)
#     im_sub_list = np.array_split(im, batch_count)
#     scores = [jaccard_sim(torch.Tensor(im_sub), torch.Tensor(s)) for im_sub in im_sub_list]
#     return torch.cat(scores, axis=0).numpy()


def jaccard_sim_batch(im, s, max_mem=2147483648, test_opt=None):
    batch_count = math.ceil(im.shape[0]*s.shape[0]*im.shape[1]*4/max_mem)
    im_sub_list = np.array_split(im, batch_count)
    scores = [jaccard_sim_transform(torch.Tensor(im_sub), torch.Tensor(s), test_opt=test_opt) for im_sub in im_sub_list]
    return torch.cat(scores, axis=0).numpy()

# def cosine_sim_batch(im, s, max_mem=2147483648, test_opt=None):
#     batch_count = math.ceil(im.shape[0]*s.shape[0]*im.shape[1]*4/max_mem)
#     im_sub_list = np.array_split(im, batch_count)
#     scores = [cosine_sim_transform(torch.Tensor(im_sub), torch.Tensor(s), test_opt=test_opt) for im_sub in im_sub_list]
#     return torch.cat(scores, axis=0).numpy()


def cal_error(videos, captions, measure='cosine', max_mem=2147483648, batch_size=1000, test_opt=None):
    if measure == 'cosine':
        if test_opt is not None:
            videos = test_opt.scale * np.subtract(videos*np.power(np.absolute(videos), test_opt.power-1), test_opt.shift)
            captions = test_opt.scale * np.subtract(captions*np.power(np.absolute(captions), test_opt.power-1), test_opt.shift)
        captions = l2norm(captions)
        videos = l2norm(videos)
        if test_opt is None or test_opt.topk == captions.shape[1]:
            errors = -1*np.dot(captions, videos.T)
        else:
            idx = 0
            errors = None
            while 1:
                sub_captions = captions[idx*batch_size:(idx+1)*batch_size,:]
                sub_captions = torch.Tensor(sub_captions)
                videos = torch.Tensor(videos)
                
                ncaps = sub_captions.shape[0]
                nvids = videos.shape[0]

                vids = videos.unsqueeze(1).expand(-1,ncaps,-1)
                caps = sub_captions.unsqueeze(0).expand(nvids,-1,-1)

                elem = torch.abs(caps*vids)

                sort_val, idx_val = torch.sort(elem, descending=True)
                topk_val = sort_val[:, :, :test_opt.topk]
                sub_errors = (-1 * (topk_val.sum(-1))).numpy()
                sub_errors = sub_errors.T
                if errors is None:
                    errors = sub_errors
                else:
                    errors = np.append(errors, sub_errors, axis=0)
                if (idx+1)*batch_size > captions.shape[0]:
                    break
                idx=idx+1
    elif measure == 'euclidean':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1':
        errors = distance.cdist(captions, videos, 'minkowski', p=1)
    elif measure == 'l2':
        errors = distance.cdist(captions, videos, 'euclidean')
    elif measure == 'l1_norm':
        errors = - distance.cdist(captions, videos, 'minkowski', p=1)/videos.shape[1]-1
    elif measure == 'l2_norm':
        errors = - distance.cdist(captions, videos, 'euclidean')/videos.shape[1]-1
    elif measure == 'jaccard':
        errors = -1*jaccard_sim_batch(captions, videos, max_mem=max_mem, test_opt=test_opt)
    return errors

# def cal_error_batch(videos, captions, measure='cosine', max_mem=2147483648, test_opt=None):
#     if measure == 'cosine':
#         captions = l2norm(captions)
#         videos = l2norm(videos)
#         errors = -1*cosine_sim_batch(captions, videos, max_mem=max_mem, test_opt=test_opt)
#     elif measure == 'euclidean':
#         errors = distance.cdist(captions, videos, 'euclidean')
#     elif measure == 'l1':
#         errors = distance.cdist(captions, videos, 'minkowski', p=1)
#     elif measure == 'l2':
#         errors = distance.cdist(captions, videos, 'euclidean')
#     elif measure == 'l1_norm':
#         errors = - distance.cdist(captions, videos, 'minkowski', p=1)/videos.shape[1]-1
#     elif measure == 'l2_norm':
#         errors = - distance.cdist(captions, videos, 'euclidean')/videos.shape[1]-1
#     elif measure == 'jaccard':
#         errors = -1*jaccard_sim_batch(captions, videos, max_mem=max_mem, test_opt=test_opt)
#     return errors


def cal_simi(captions, videos, measure='cosine'):
    if measure == 'cosine':
        captions = l2norm(captions)
        videos = l2norm(videos)
        errors = np.dot(captions, videos.T)
    elif measure == 'jaccard':
        errors = jaccard_sim_batch(captions, videos)
    return errors


# predict tags
def pred_tag(tag_prob_embs, video_ids, tag_vocab_path, tag_vocab_size, output_dir, k=10):
    all_vocab_list = np.loadtxt(os.path.join(tag_vocab_path, "all_words.txt"), dtype=str)[:, 0]
    tag_vocab_list = all_vocab_list[:tag_vocab_size].tolist()
    tag_vocab_size = len(tag_vocab_list)
    idx2tag = dict(zip(range(tag_vocab_size), tag_vocab_list))
    # print(tag_prob_embs.shape)
    assert tag_prob_embs.shape[1] == tag_vocab_size, "%s != %s" % (tag_prob_embs.shape[1], tag_vocab_size)
    
    output_file = os.path.join(output_dir, 'pred_tags.txt')
    makedirsforfile(output_file)
    fout = open(output_file, 'w')

    for idx, prob_vec in enumerate(tag_prob_embs):
        vid_id = video_ids[idx]
        top_hits = np.argsort(prob_vec)[::-1][:k]
        fout.write(vid_id + '\t')
        for hit in top_hits:
            fout.write("%s:%.3f " % (idx2tag[hit], prob_vec[hit]))
        fout.write('\n')
    fout.close()


# encode text or video, latent or concept or both spaces
def encode(encoder, data_loader, return_ids=True):
    """Encode all videos and captions loadable by `data_loader`
    """
    # numpy array to keep all the embeddings
    init_flag = True
    ids = ['']*len(data_loader.dataset)
    pbar = Progbar(len(data_loader.dataset))
    for i, (datas, idxs, data_ids) in enumerate(data_loader):

        # compute the embeddings
        batch_embs = encoder(datas)

        # initialize the numpy arrays given the size of the embeddings
        if init_flag:
            init_flag = False
            l = len(data_loader.dataset)
            embs = [np.zeros((l, space_embs.size(1))) if space_embs is not None else None for space_embs in batch_embs]

        # preserve the embeddings by copying from gpu and converting to numpy
        for s, space_embs in enumerate(batch_embs):
            if space_embs is not None:
                embs[s][idxs] = space_embs.data.cpu().numpy().copy()

        for j, idx in enumerate(idxs):
            ids[idx] = data_ids[j]

        del datas
        pbar.add(len(idxs))

    return embs, ids if return_ids == True else embs
