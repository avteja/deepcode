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
from transformer.Optim import ScheduledOptim
import os
import json

import sentencepiece as spm

from transformer.Translator import Translator

sp = spm.SentencePieceProcessor()

########################################
# Warm restart code credits: Ankur Garg#
# First four functions in this code   #
########################################
def check_restart_conditions(opt):
    # Check for the status file corresponding to the model
    status_file = os.path.join(opt.save_model_dir, 'status.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
        opt.resume_from_epoch = status['epoch']
    else:
        opt.resume_from_epoch = 0
    return opt

def write_status(opt, epoch):
    status_file = os.path.join(opt.save_model_dir, 'status.json')
    status = {
        'epoch': epoch,
    }
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=4)

def load_models(model, opt, epoch):
    checkpoint = torch.load(os.path.join(opt.save_model_dir, 'epoch_{0:02d}.chkpt'.format(epoch)))
    model.load_state_dict(checkpoint['model'])
    return model

def save_params(opt):
    if not os.path.exists(os.path.join(opt.save_model_dir)):
        os.makedirs(os.path.join(opt.save_model_dir))

    with open(os.path.join(opt.save_model_dir, 'params.json'), 'w') as f:
        json.dump(vars(opt), f, indent=4)

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


def train_epoch(model, training_data, optimizer, device, smoothing, loss_weight = 1):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        #w = input("s")
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)

        loss *= loss_weight

        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct
        # break

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            #print(src_seq,src_pos,tgt_seq,tgt_seq)
            batch_size = src_seq.size(0)

            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_bleu_score(opt, model, data, device, epoch, split = 'dev'):
    translator = Translator(opt, model, load_from_file = False)
    hyp_file = os.path.join(opt.save_model_dir, 'mypreds' + str(epoch) + '.hyp')
    outfile = open(hyp_file, 'w')
    for batch in tqdm(data, mininterval=2, desc='  - (Test)', leave=False):
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        all_hyp, all_scores = translator.translate_batch(src_seq, src_pos)
        for idx_seqs in all_hyp:
            for idx_seq in idx_seqs:
                pred = idx_seq
                out = sp.DecodeIds(pred)
                outfile.write(out + '\n')
    outfile.close()
    os.system("sh calcBLEU.sh " + split + " " + opt.save_model_dir + " " + str(epoch))

def train(model, training_data, validation_data, test_data, mined_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = os.path.join(opt.save_model_dir, 'train.log')
        log_valid_file = os.path.join(opt.save_model_dir, 'valid.log')

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.resume_from_epoch, opt.resume_from_epoch + opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Train)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                   accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        if opt.loss_weight > 0:
            mined_loss, mined_accu = train_epoch(
                model, mined_data, optimizer, device, smoothing=opt.label_smoothing, loss_weight = opt.loss_weight)
            print('  - (Mined)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
                  'elapse: {elapse:3.3f} min'.format(loss=mined_loss,
                      accu=100*mined_accu,
                      elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Valid)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                    accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        if (epoch_i+1)%(opt.test_epoch) == 0:
            eval_bleu_score(opt, model, test_data, device, epoch_i, split = 'test')

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }

        if opt.save_model_dir:
            if opt.save_mode == 'all':
                model_name = os.path.join(opt.save_model_dir, 'epoch_{0:02d}.chkpt'.format(epoch_i))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = os.path.join(opt.save_model_dir, 'epoch_{0:02d}.chkpt'.format(epoch_i))
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

        write_status(opt, epoch_i)

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-mined_data', required=True)
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

    parser.add_argument('-log', type=bool, default=True)
    parser.add_argument('-save_model_dir', default=None, required=True)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    # For bleu eval
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")

    parser.add_argument('-test_epoch', type=int, default=5, help='Test every x epochs')
    parser.add_argument('-resume_from_epoch', type=int, default=0, help='Warm restart')

    # Not really needed
    parser.add_argument('-alpha', type=float,default=1.0, help='Weighting loss')
    parser.add_argument('-loss_weight', type=float,default=0.1, help='Mined loss weight')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # Snippet model sentencepiece
    sp.Load(opt.snippet_model)

    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    mined_data = torch.load(opt.mined_data)

    opt.inp_seq_max_len = 4*data['settings'].train_max_input_len
    opt.out_seq_max_len = 4*data['settings'].train_max_output_len
    
    opt.max_token_seq_len = int(opt.out_seq_max_len/4)

    training_data, validation_data, test_data, mined_data = prepare_dataloaders(data, mined_data, opt)

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

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    save_params(opt)

    opt = check_restart_conditions(opt)
    if opt.resume_from_epoch >= 1:
        print('Loading Old model')
        print('Loading model files from folder: %s' % opt.save_model_dir)
        transformer = load_models(transformer, opt, opt.resume_from_epoch)

    train(transformer, training_data, validation_data, test_data, mined_data, optimizer, device, opt)

def prepare_dataloaders(data, mined_data, opt):
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

    mined_loader = torch.utils.data.DataLoader(
            TranslationDataset(
                src_word2idx=mined_data['dict']['src'],
                tgt_word2idx=mined_data['dict']['tgt'],
                src_insts=mined_data['train']['src'],
                tgt_insts=mined_data['train']['tgt']),
            num_workers=2,
            batch_size=opt.batch_size,
            collate_fn=paired_collate_fn)

    return train_loader, valid_loader, test_loader, mined_loader




if __name__ == '__main__':
    main()
