import logging
import numpy as np
import evaluation
import util.metrics as metrics
import torch


def norm_score(t2v_all_errors):
    t2v_all_score = -t2v_all_errors
    t2v_all_score = t2v_all_score - np.min(t2v_all_score)
    t2v_all_score = t2v_all_score / np.max(t2v_all_score)
    return -t2v_all_score


def get_mod_embeddings(data_loader, model_embed):
    # compute encoding and mapping for all validation or test samples for one modality
    # space 0: latent, space 1: concept, both: hybrid
    with torch.no_grad():
        embs, ids = evaluation.encode(model_embed, data_loader)
    return embs, ids


def get_embeddings(vid_data_loader, txt_data_loader, model):
    # compute the encoding and mapping for all the validation video and captions
    model.val_start()
    with torch.no_grad():
        vid_embs, vid_ids = evaluation.encode(model.embed_vis, vid_data_loader)
        txt_embs, txt_ids = evaluation.encode(model.embed_txt, txt_data_loader)
    return [vid_embs, txt_embs], [vid_ids, txt_ids]


def cal_perf(t2v_all_errors, v2t_gt, t2v_gt, tb_logger=None, Eiters=0, space=None):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr) = metrics.eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = metrics.t2v_map(t2v_all_errors, t2v_gt)

    # caption retrieval
    (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr) = metrics.eval_q2m(t2v_all_errors.T, v2t_gt)
    v2t_map_score = metrics.v2t_map(t2v_all_errors, v2t_gt)

    if space is not None:
        logging.info(space.title() + " Space Evaluation")

    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10, medr, meanr: {}".format([round(t2v_r1, 3), round(t2v_r5, 3), round(t2v_r10, 3),
                                                        round(t2v_medr, 3), round(t2v_meanr, 3)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10, 3)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    logging.info(" * Video to text:")
    logging.info(" * r_1_5_10, medr, meanr: {}".format([round(v2t_r1, 3), round(v2t_r5, 3), round(v2t_r10, 3),
                                                        round(v2t_medr, 3), round(v2t_meanr, 3)]))
    logging.info(" * recall sum: {}".format(round(v2t_r1+v2t_r5+v2t_r10, 3)))
    logging.info(" * mAP: {}".format(round(v2t_map_score, 4)))
    logging.info(" * "+'-'*10)

    if tb_logger is not None:        
        # record metrics in tensorboard
        tb_logger.log_value('v2t_r1', v2t_r1, step=Eiters)
        tb_logger.log_value('v2t_r5', v2t_r5, step=Eiters)
        tb_logger.log_value('v2t_r10', v2t_r10, step=Eiters)
        tb_logger.log_value('v2t_medr', v2t_medr, step=Eiters)
        tb_logger.log_value('v2t_meanr', v2t_meanr, step=Eiters)
        tb_logger.log_value('v2t_map', v2t_map_score, step=Eiters)

        tb_logger.log_value('t2v_r1', t2v_r1, step=Eiters)
        tb_logger.log_value('t2v_r5', t2v_r5, step=Eiters)
        tb_logger.log_value('t2v_r10', t2v_r10, step=Eiters)
        tb_logger.log_value('t2v_medr', t2v_medr, step=Eiters)
        tb_logger.log_value('t2v_meanr', t2v_meanr, step=Eiters)
        tb_logger.log_value('t2v_map', t2v_map_score, step=Eiters)

    v2t = (v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr, v2t_map_score)
    t2v = (t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr, t2v_map_score)
    return v2t, t2v


def get_metrics(opt, embs, ids, tb_logger=None, Eiters=0, w=0.6, pred_error_matrix_file=None, test_opt=None):
    # Get "groud truth".
    video_ids, caption_ids = ids
    v2t_gt, t2v_gt = metrics.get_gt(video_ids, caption_ids)

    # Get error matrices and performance metrics.
    vid_embs, txt_embs = embs
    if opt.space in ['latent', 'hybrid']:
        t2v_all_errors_1 = norm_score(evaluation.cal_error(vid_embs[0], txt_embs[0], opt.measure, test_opt=None))
        v2t, t2v = cal_perf(t2v_all_errors_1, v2t_gt, t2v_gt, tb_logger=tb_logger, Eiters=Eiters, space='latent')

    if opt.space in ['concept', 'hybrid']:
        t2v_all_errors_2 = norm_score(evaluation.cal_error(vid_embs[1], txt_embs[1], opt.measure_2, test_opt=test_opt))
        v2t, t2v = cal_perf(t2v_all_errors_2, v2t_gt, t2v_gt, tb_logger=tb_logger, Eiters=Eiters, space='concept')

    if opt.space in ['hybrid']:
        t2v_all_errors = w * t2v_all_errors_1 + (1-w) * t2v_all_errors_2
        v2t, t2v = cal_perf(t2v_all_errors, v2t_gt, t2v_gt, tb_logger=tb_logger, Eiters=Eiters, space='hybrid')

    if pred_error_matrix_file is not None:
        torch.save({'errors': t2v_all_errors, 'videos': video_ids, 'captions': caption_ids}, pred_error_matrix_file)    
        logging.info("write prediction error matrix into: %s" % pred_error_matrix_file)

    return v2t, t2v


def validate(opt, tb_logger, vid_data_loader, text_data_loader, model):
    # Compute the encoding and mapping for all the validation video and captions
    embs, ids = get_embeddings(vid_data_loader, text_data_loader, model)

    # Get error matrices and performance metrics.
    v2t, t2v = get_metrics(opt, embs, ids, tb_logger=tb_logger, Eiters=model.Eiters)
    
    # Compute the validation score
    v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr, v2t_map_score = v2t
    t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr, t2v_map_score = t2v
    currscore = 0
    if opt.val_metric == "recall":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += (v2t_r1 + v2t_r5 + v2t_r10)
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += (t2v_r1 + t2v_r5 + t2v_r10)
    elif opt.val_metric == "map":
        if opt.direction == 'i2t' or opt.direction == 'all':
            currscore += v2t_map_score
        if opt.direction == 't2i' or opt.direction == 'all':
            currscore += t2v_map_score

    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore
