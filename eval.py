'''
This script handles the training process.
'''

import argparse
import dill as pickle
import torch.nn.functional as F
from tqdm import tqdm
import math
import torch
from torchtext.legacy.data import Dataset, BucketIterator
from torchtext.legacy.datasets import TranslationDataset
#import nltk

import transformer.Constants as Constants
from transformer.Models import Transformer

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss



def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold



def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = patch_src(batch.src, opt.src_pad_idx).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, opt.trg_pad_idx))

            # forward
            pred = model(src_seq, trg_seq)
            loss, n_correct, n_word = cal_performance(
                pred, gold, opt.trg_pad_idx, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy




def main():
    '''
    Usage:
python eval.py \
-model output/wmt16_n/model.chkpt \
-data_pkl m30k_deen_shr.pkl \
-label_smoothing \
-proj_share_weight \
-scale_emb_or_prj emb \
-b 512 
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-model', required=True, help='Path to model weight file')
    parser.add_argument('-data_pkl', default=None)     # all-in-1 data pickle or bpe field
    parser.add_argument('-test_path', default=None)     # bpe encoded data
    parser.add_argument('-b', '--batch_size', type=int, default=4)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    
    parser.add_argument('-h_attn', action='store_true')
    parser.add_argument('-v_attn', action='store_true')
    
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    device = torch.device('cuda' if opt.cuda else 'cpu')

    #========= Loading Dataset =========#

    if opt.test_path: # WMT17 dataset
        test_data = prepare_dataloaders_from_bpe_files(opt, device)
    elif opt.data_pkl: # WMT16 dataset
        test_data = prepare_dataloaders(opt, device)
    else: 
        raise
        

    print(opt)

    model = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        h_attn=opt.h_attn,
        v_attn=opt.v_attn,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)
    
    print('Using horizontal attention: %s' % opt.h_attn)
    print('Using vertical attention: %s' % opt.v_attn)

    checkpoint = torch.load(opt.model, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    valid_loss, accuracy = eval_epoch(model, test_data, device, opt)

    
    valid_ppl = math.exp(min(valid_loss, 100))
    print('PPL: %f' % valid_ppl)


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = data['settings'].max_len
    field = data['vocab']
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    test = TranslationDataset(
        fields=fields,
        path=opt.test_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)
    test_iterator = BucketIterator(test, batch_size=batch_size, device=device)
    return test_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg':data['vocab']['trg']}

    
    test = Dataset(examples=data['test'], fields=fields)
    
    test_iterator = BucketIterator(test, batch_size=batch_size, device=device)

    return test_iterator




if __name__ == '__main__':
    main()
    #profile()
