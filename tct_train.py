'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import Transformer
from transformer.NewModels import Transformer2
from transformer.Optim import ScheduledOptim
import os

import sentencepiece as spm

from transformer.Translator import Translator

sp = spm.SentencePieceProcessor()

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def print_preds(pred, batch_size, outfile):
    pred = pred.view(batch_size,-1,pred.size(1))
    pred = pred.max(2)[1]

    for b in range(batch_size):
        sent = pred[b,:]
        indices = []
        for token in sent:
            if token != Constants.EOS:
                indices.append(token.item())
            else:
                break
        out = sp.DecodeIds(indices)
        outfile.write(out + '\n')

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, model2, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''
    #TODO
    model.train()
    model2.train()

    total_code_loss = 0
    n_code_word_total = 0
    n_code_word_correct = 0

    total_text_loss = 0
    n_text_word_total = 0
    n_text_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold_code = tgt_seq[:, 1:]
        gold_text = src_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        #w = input("s")
        pred, dec_output, new_enc_slf_attn_mask, new_dec_slf_attn_mask, new_dec_subsequent_mask, new_enc_non_pad_mask, new_dec_non_pad_mask, new_dec_enc_attn_mask = model(src_seq, src_pos, tgt_seq, tgt_pos, return_masks = True)

        pred2 = model2(dec_output, tgt_pos, src_seq, src_pos, new_enc_slf_attn_mask, new_dec_slf_attn_mask, new_dec_subsequent_mask, new_enc_non_pad_mask, new_dec_non_pad_mask, new_dec_enc_attn_mask)
        
        # backward
        code_loss, n_code_correct = cal_performance(pred, gold_code, smoothing=smoothing)
        text_loss, n_text_correct = cal_performance(pred2, gold_text, smoothing = smoothing)
        
        (code_loss + text_loss).backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_code_loss += code_loss.item()
        total_text_loss += text_loss.item()

        non_pad_mask = gold_code.ne(Constants.PAD)
        n_code_word = non_pad_mask.sum().item()
        n_code_word_total += n_code_word
        n_code_word_correct += n_code_correct

        non_pad_mask = gold_text.ne(Constants.PAD)
        n_text_word = non_pad_mask.sum().item()
        n_text_word_total += n_text_word
        n_text_word_correct += n_text_correct
        
    code_loss_per_word = total_code_loss/n_code_word_total
    text_loss_per_word = total_text_loss/n_text_word_total
    code_accuracy = n_code_word_correct/n_code_word_total
    text_accuracy = n_text_word_correct/n_text_word_total
    return code_loss_per_word, text_loss_per_word, code_accuracy, text_accuracy

def eval_epoch(model, model2, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    model2.eval()
    
    total_code_loss = 0
    n_code_word_total = 0
    n_code_word_correct = 0

    total_text_loss = 0
    n_text_word_total = 0
    n_text_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold_code = tgt_seq[:, 1:]
            gold_text = src_seq[:, 1:]

            # forward
            pred, dec_output, new_enc_slf_attn_mask, new_dec_slf_attn_mask, new_dec_subsequent_mask, new_enc_non_pad_mask, new_dec_non_pad_mask, new_dec_enc_attn_mask = model(src_seq, src_pos, tgt_seq, tgt_pos, return_masks = True)

            pred2 = model2(dec_output, tgt_pos, src_seq, src_pos, new_enc_slf_attn_mask, new_dec_slf_attn_mask, new_dec_subsequent_mask, new_enc_non_pad_mask, new_dec_non_pad_mask, new_dec_enc_attn_mask)
            
            # loss
            code_loss, n_code_correct = cal_performance(pred, gold_code)

            text_loss, n_text_correct = cal_performance(pred2, gold_text)


            # note keeping
            total_code_loss += code_loss.item()
            total_text_loss += text_loss.item()

            non_pad_mask = gold_code.ne(Constants.PAD)
            n_code_word = non_pad_mask.sum().item()
            n_code_word_total += n_code_word
            n_code_word_correct += n_code_correct

            non_pad_mask = gold_text.ne(Constants.PAD)
            n_text_word = non_pad_mask.sum().item()
            n_text_word_total += n_text_word
            n_text_word_correct += n_text_correct

    code_loss_per_word = total_code_loss/n_code_word_total
    text_loss_per_word = total_text_loss/n_text_word_total
    code_accuracy = n_code_word_correct/n_code_word_total
    text_accuracy = n_text_word_correct/n_text_word_total
    return code_loss_per_word, text_loss_per_word, code_accuracy, text_accuracy

def eval_bleu_score(opt, model, data, device, split = 'dev'):
    translator = Translator(opt, model, load_from_file = False)
    outfile = open('results/mypreds.hyp', 'w')
    for batch in tqdm(data, mininterval=2, desc='  - (Validation)', leave=False):
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = translator.translate_batch(src_seq, src_pos)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred = idx_seq
                out = sp.DecodeIds(pred)
                outfile.write(out + '\n')
    outfile.close()
    os.system("sh calcBLEU.sh " + split)

def train(model, model2, training_data, validation_data, test_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_code_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_code_loss, train_text_loss, train_code_accu, train_text_accu = train_epoch(
            model, model2, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   code_loss: {code_loss: 8.5f}, code_ppl: {code_ppl: 8.5f}, code_accuracy: {code_accu:3.3f} %, text_loss: {text_loss: 8.5f}, text_ppl: {text_ppl: 8.5f}, text_accuracy: {text_accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  code_loss=train_code_loss, code_ppl=math.exp(min(train_code_loss, 100)), code_accu=100*train_code_accu, text_loss=train_text_loss, text_ppl=math.exp(min(train_text_loss, 100)), text_accu=100*train_text_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_code_loss, valid_text_loss, valid_code_accu, valid_text_accu = eval_epoch(
            model, model2, validation_data, device)
        print('  - (Validation)   code_loss: {code_loss: 8.5f}, code_ppl: {code_ppl: 8.5f}, code_accuracy: {code_accu:3.3f} %, text_loss: {text_loss: 8.5f}, text_ppl: {text_ppl: 8.5f}, text_accuracy: {text_accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  code_loss=valid_code_loss, code_ppl=math.exp(min(valid_code_loss, 100)), code_accu=100*valid_code_accu, text_loss=valid_text_loss, text_ppl=math.exp(min(valid_text_loss, 100)), text_accu=100*valid_text_accu,
                  elapse=(time.time()-start)/60))

        valid_code_accus += [valid_code_accu]

        if (epoch_i+1)%10 == 0:
            eval_bleu_score(opt, model, test_data, device, split = 'test')

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_code_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_code_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_code_accu >= max(valid_code_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{code_loss: 8.5f},{code_ppl: 8.5f},{code_accu:3.3f},{text_loss: 8.5f},{text_ppl: 8.5f},{text_accu:3.3f}\n'.format(
                    epoch=epoch_i, code_loss=train_code_loss,
                    code_ppl=math.exp(min(train_code_loss, 100)), code_accu=100*train_code_accu, text_loss=train_text_loss,
                    text_ppl=math.exp(min(train_text_loss, 100)), text_accu=100*train_text_accu))
                log_vf.write('{epoch},{code_loss: 8.5f},{code_ppl: 8.5f},{code_accu:3.3f},{text_loss: 8.5f},{text_ppl: 8.5f},{text_accu:3.3f}\n'.format(
                    epoch=epoch_i, code_loss=valid_code_loss,
                    code_ppl=math.exp(min(valid_code_loss, 100)), code_accu=100*valid_code_accu, text_loss=valid_text_loss,
                    text_ppl=math.exp(min(valid_text_loss, 100)), text_accu=100*valid_text_accu))
                
def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-snippet_model', required=True)

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    # For bleu eval
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")



    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # Snippet model sentencepiece
    sp.Load(opt.snippet_model)

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.inp_seq_max_len = 4*data['settings'].train_max_input_len
    opt.out_seq_max_len = 4*data['settings'].train_max_output_len
    
    opt.max_token_seq_len = int(opt.out_seq_max_len/4)

    training_data, validation_data, test_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    print(opt.inp_seq_max_len,opt.out_seq_max_len,opt.src_vocab_size,opt.tgt_vocab_size)

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.inp_seq_max_len,
        opt.out_seq_max_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)
    
    transformer2 = Transformer2(
            opt.src_vocab_size,
            opt.out_seq_max_len,
            opt.inp_seq_max_len,
            tgt_emb_prj_weight_sharing=opt.proj_share_weight,
            emb_src_tgt_weight_sharing=opt.embs_share_weight,
            d_k=opt.d_k,
            d_v=opt.d_v,
            d_model=opt.d_model,
            d_word_vec=opt.d_word_vec,
            d_inner=opt.d_inner_hid,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            dropout=opt.dropout).to(device)

    #optimizer_params_group = [ { 'params': transformer.parameters()},{'params': transformer2.parameters()} ]
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, list(transformer.parameters())+list(transformer2.parameters())),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, transformer2, training_data, validation_data, test_data, optimizer, device ,opt)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['test']['src'],
            tgt_insts=data['test']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    main()
