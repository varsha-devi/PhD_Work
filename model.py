import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from loss import TripletLoss
from basic.bigfile import BigFile
from collections import OrderedDict


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we)) 
    return np.array(we)


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


class MFC(nn.Module):
    """
    Multi Fully Connected Layers, currently working with just one
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=True, have_last_bn=True, xavier_init=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights(xavier_init)

    def init_weights(self, xavier_init):
        """Xavier initialization for the fully connected layer
        """
        if xavier_init and self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch normalization
            if self.have_bn and self.have_last_bn:
                features = self.bn1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features


class Video_multilevel_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        self.tag_vocab_size = opt.tag_vocab_size

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.visual_kernel_sizes
                ])

    def forward(self, videos):
        """Extract video feature vectors."""
        videos, videos_origin, lengths, videos_mask = videos
        
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        if self.gru_pool == 'mean':
            mean_gru = torch.zeros(gru_init_out.size(0), self.rnn_output_size).cuda()
            for i, batch in enumerate(gru_init_out):
                mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
            gru_out = mean_gru
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, videos_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        videos_mask = videos_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * videos_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full':
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)


class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.text_kernel_sizes
                ])

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors
        cap_wids, cap_bows, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
    
        if self.gru_pool == 'mean':
            gru_out = torch.zeros(padded[0].size(0), self.rnn_output_size).cuda()
            for i, batch in enumerate(padded[0]):
                gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, cap_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced': # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features


class Mapping(nn.Module):
    """
    Section 4.1. Hybrid space mapping (sigmoid and l2norm removed, moved to loss function)
    """
    def __init__(self, latent_mapping_layers, dropout, tag_vocab_size, latent_xavier_init=True, concept_xavier_init=False,
                 latent_have_dp=True, concept_have_dp=False, concept_transform=False):
        super(Mapping, self).__init__()
        concept_mapping_layers = latent_mapping_layers.copy()
        concept_mapping_layers[1] = tag_vocab_size
        self.latent_mapping = MFC(latent_mapping_layers, dropout, xavier_init=latent_xavier_init, have_dp=latent_have_dp) \
                              if latent_mapping_layers[1] else None
        self.concept_mapping = MFC(concept_mapping_layers, dropout, xavier_init=concept_xavier_init, have_dp=concept_have_dp) \
                              if concept_mapping_layers[1] else None
        self.second_mapping = None if concept_transform else None

    def forward(self, features):
        lat_embs = self.latent_mapping(features) if self.latent_mapping is not None else None
        con_embs = self.concept_mapping(features) if self.concept_mapping is not None else None
        sec_embs = self.second_mapping(con_embs) if self.second_mapping is not None else None
        return [lat_embs, con_embs, sec_embs]


class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict(),
                      self.vid_mapping.state_dict(), self.text_mapping.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])
        self.vid_mapping.load_state_dict(state_dict[2])
        self.text_mapping.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()
        self.vid_mapping.train()
        self.text_mapping.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()
        self.vid_mapping.eval()
        self.text_mapping.eval()

    def init_info(self):

        # init gpu
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            self.vid_mapping.cuda()
            self.text_mapping.cuda()
            cudnn.benchmark = True

        # init params
        params = list(self.vid_encoding.parameters())
        params += list(self.text_encoding.parameters())
        params += list(self.vid_mapping.parameters())
        params += list(self.text_mapping.parameters())
        self.params = params

        # print structure
        print(self.vid_encoding)
        print(self.text_encoding)
        print(self.vid_mapping)
        print(self.text_mapping)


class Dual_Encoding_Multi_Space(BaseModel):
    """
    dual encoding network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.tag_vocab_size = opt.tag_vocab_size
        self.measure_2 = opt.measure_2
        self.space = opt.space

        if hasattr(opt, 'classification_loss_weight'):
            self.classification_loss_weight = opt.classification_loss_weight
        else:
            self.classification_loss_weight = 1.0

        if hasattr(opt, 'latent_no_dp'):
            latent_have_dp = not opt.latent_no_dp
        else:
            latent_have_dp = True

        if hasattr(opt, 'latent_no_xi'):
            latent_xavier_init = not opt.latent_no_xi
        else:
            latent_xavier_init = True

        if hasattr(opt, 'concept_xi'):
            concept_xavier_init = opt.concept_xi
        else:
            concept_xavier_init = False

        if hasattr(opt, 'concept_dp'):
            concept_have_dp = opt.concept_dp
        else:
            concept_have_dp = False

        self.vid_encoding = Video_multilevel_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)

        self.vid_mapping = Mapping(opt.visual_mapping_layers, opt.dropout, opt.tag_vocab_size,
                                   latent_xavier_init=latent_xavier_init, concept_xavier_init=concept_xavier_init,
                                   latent_have_dp=latent_have_dp, concept_have_dp=concept_have_dp)
        self.text_mapping = Mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size,
                                    latent_xavier_init=latent_xavier_init, concept_xavier_init=concept_xavier_init,
                                    latent_have_dp=latent_have_dp, concept_have_dp=concept_have_dp)

        self.init_info()

        # Loss and Optimizer
        self.triplet_latent_criterion = TripletLoss(margin=opt.margin,
                                        measure=opt.measure,
                                        max_violation=opt.max_violation,
                                        cost_style=opt.cost_style,
                                        direction=opt.direction)
        self.triplet_concept_criterion = TripletLoss(margin=opt.margin_2,
                                        measure=opt.measure_2,
                                        max_violation=opt.max_violation,
                                        cost_style=opt.cost_style,
                                        direction=opt.direction)
        self.tag_criterion = nn.BCEWithLogitsLoss()

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.Eiters = 0

    def embed_vis(self, vis_data):
        """Compute the video embeddings
        """
        # video data
        frames, mean_origin, video_lengths, videos_mask = vis_data
        if torch.cuda.is_available():
            frames = frames.cuda()
            mean_origin = mean_origin.cuda()
            videos_mask = videos_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, videos_mask)

        return self.vid_mapping(self.vid_encoding(vis_data))

    def embed_txt(self, txt_data):
        """Compute the caption embeddings
        """
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if torch.cuda.is_available():
            if captions is not None:
                captions = captions.cuda()
            if cap_bows is not None:
                cap_bows = cap_bows.cuda()
            if cap_masks is not None:
                cap_masks = cap_masks.cuda()
        txt_data = (captions, cap_bows, lengths, cap_masks)

        return self.text_mapping(self.text_encoding(txt_data))

    def forward_loss(self, embs, target_tag, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        # classification on both video and text 
        # (vid_emb, vid_tag_score), (cap_emb, cap_tag_score) = embs
        vid_embs, cap_embs = embs
        if cap_embs[0] is not None:
            batch_size = cap_embs[0].shape[0]
        else:
            batch_size = cap_embs[1].shape[0]

        loss_1 = self.triplet_latent_criterion(cap_embs[0], vid_embs[0]) if vid_embs[0] is not None else 0.0
        if vid_embs[2] is not None:
            loss_2 = self.triplet_concept_criterion(cap_embs[2], vid_embs[2])
        elif vid_embs[1] is not None:
            loss_2 = self.triplet_concept_criterion(cap_embs[1], vid_embs[1])
        else: 0.0

        loss_3 = self.tag_criterion(vid_embs[1], target_tag) if vid_embs[1] is not None else 0.0
        loss_4 = self.tag_criterion(cap_embs[1], target_tag) if cap_embs[1] is not None else 0.0
        
        loss = loss_1 + loss_2 + self.classification_loss_weight * batch_size * (loss_3 + loss_4)
        if vid_embs[0] is not None:
            self.logger.update('Le', loss.item(), vid_embs[0].size(0))
        else:
            self.logger.update('Le', loss.item(), vid_embs[1].size(0))

        return loss

    def train_emb(self, videos, captions, target_tag, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        embs = self.embed_vis(videos), self.embed_txt(captions)

        if torch.cuda.is_available():
            target_tag = target_tag.cuda()

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(embs, target_tag)
        loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        if embs[0][0] is not None:
            batch_size = embs[0][0].size(0)
        else:
            batch_size = embs[0][1].size(0)
        return batch_size, loss_value

    # def get_pre_tag(self, vid_emb_wo_norm):
    #     pred_score = vid_emb_wo_norm[:,:self.tag_vocab_size]
    #     pred_prob = torch.sigmoid(pred_scoreb)
    #     return pred_prob


NAME_TO_MODELS = {'dual_encoding_latent': Dual_Encoding_Multi_Space, 'dual_encoding_hybrid': Dual_Encoding_Multi_Space , \
                  'dual_encoding_concept': Dual_Encoding_Multi_Space}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
