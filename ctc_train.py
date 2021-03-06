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
from transformer.GaussianNoise import GaussianNoise
import os
import json

import sentencepiece as spm

from transformer.Translator import Translator

sp = spm.SentencePieceProcessor()

########################################
# Warm restart code credits: Ankur Garg#
# First four functions in this code    #
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

def load_models(modelTC, modelCT, opt, epoch):
    checkpoint = torch.load(os.path.join(opt.save_model_dir, 'epoch_{0:02d}.chkpt'.format(epoch)))
    modelTC.load_state_dict(checkpoint['modelTC'])
    modelCT.load_state_dict(checkpoint['modelCT'])
    return modelTC, modelCT

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


def train_epoch(modelTC, modelCT, training_data, mined_data, optimizer, device, smoothing, opt, noise_model = None):
    ''' Epoch operation in training phase'''
    modelTC.train()
    modelCT.train()

    total_code_loss = 0
    n_code_word_total = 0
    n_code_word_correct = 0

    total_recon_loss = 0
    n_recon_word_total = 0
    n_recon_word_correct = 0

    for train_batch, mined_batch in tqdm(zip(training_data, mined_data), mininterval=2,
            desc='  - (Training)   ', leave=False, total=len(training_data)):

        optimizer.zero_grad()
  
        ## Train Data - TC Only
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), train_batch)
        gold_code = tgt_seq[:, 1:]
        gold_text = src_seq[:, 1:]

        
        pred_code, dec_output = modelTC(src_seq, src_pos, tgt_seq, tgt_pos, input_logits = False) 

        # Loss
        code_loss, n_code_correct = cal_performance(pred_code, gold_code, smoothing=smoothing)
        
        ## Mined Data - CTC Reconstruction
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), mined_batch)
        gold_recon = tgt_seq[:, 1:]
        # gold_text = src_seq[:, 1:]
        
        if True:

            pred_text, dec_output = modelCT(tgt_seq, tgt_pos, src_seq, src_pos, input_logits = False)
            if noise_model is not None:
            	dec_output = noise_model(dec_output)

            pred_recon = modelTC(dec_output, src_pos, tgt_seq, tgt_pos, input_logits = True, true_src_seq = src_seq) 

            # Loss
            recon_loss, n_recon_correct = cal_performance(pred_recon, gold_recon, smoothing=smoothing)
        else:
            recon_loss = torch.Tensor([0.0]).to(device)
            n_recon_correct = 0

        # Backward
        (code_loss + opt.alpha*recon_loss).backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_code_loss += code_loss.item()
        total_recon_loss += recon_loss.item()

        non_pad_mask = gold_code.ne(Constants.PAD)
        n_code_word = non_pad_mask.sum().item()
        n_code_word_total += n_code_word
        n_code_word_correct += n_code_correct

        non_pad_mask = gold_recon.ne(Constants.PAD)
        n_recon_word = non_pad_mask.sum().item()
        n_recon_word_total += n_recon_word
        n_recon_word_correct += n_recon_correct
        
    code_loss_per_word = total_code_loss/n_code_word_total
    recon_loss_per_word = total_recon_loss/n_recon_word_total
    code_accuracy = n_code_word_correct/n_code_word_total
    recon_accuracy = n_recon_word_correct/n_recon_word_total
    return code_loss_per_word, recon_loss_per_word, code_accuracy, recon_accuracy

def eval_epoch(modelTC, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    modelTC.eval()
    # modelCT.eval()
    
    total_code_loss = 0
    n_code_word_total = 0
    n_code_word_correct = 0

    # total_recon_loss = 0
    # n_recon_word_total = 0
    # n_recon_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

            ## Train Data - TC Only
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold_code = tgt_seq[:, 1:]
            gold_text = src_seq[:, 1:]

            
            pred_code, dec_output = modelTC(src_seq, src_pos, tgt_seq, tgt_pos, input_logits = False) 

            # Loss
            code_loss, n_code_correct = cal_performance(pred_code, gold_code)
            
            # ## Mined Data - CTC Reconstruction
            # src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), mined_data)
            # gold_code = tgt_seq[:, 1:]
            # gold_text = src_seq[:, 1:]

            # pred_text, dec_output = modelCT(tgt_seq, tgt_pos, src_seq, src_pos, input_logits = False)
            
            # pred_code = modelTC(dec_output, src_pos, tgt_seq, tgt_pos, , true_src_seq = src_seq) 

            # # Loss
            # recon_loss, n_recon_correct = cal_performance(pred_code, gold_code, smoothing=smoothing)

            # note keeping
            total_code_loss += code_loss.item()
            # total_recon_loss += recon_loss.item()

            non_pad_mask = gold_code.ne(Constants.PAD)
            n_code_word = non_pad_mask.sum().item()
            n_code_word_total += n_code_word
            n_code_word_correct += n_code_correct

            # non_pad_mask = gold_recon.ne(Constants.PAD)
            # n_recon_word = non_pad_mask.sum().item()
            # n_recon_word_total += n_recon_word
            # n_recon_word_correct += n_recon_correct
        
    code_loss_per_word = total_code_loss/n_code_word_total
    # recon_loss_per_word = total_recon_loss/n_recon_word_total
    code_accuracy = n_code_word_correct/n_code_word_total
    # recon_accuracy = n_recon_word_correct/n_recon_word_total
    return code_loss_per_word, code_accuracy

def eval_bleu_score(opt, modelTC, data, device, epoch, split = 'dev'):
    translator = Translator(opt, modelTC, load_from_file = False)
    hyp_file = os.path.join(opt.save_model_dir, 'mypreds_' + split + '_' + str(epoch) + '.hyp')
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

def train(modelTC, modelCT, training_data, mined_data, validation_data, test_data, optimizer, device, opt,noise_model=None):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    log_test_file = None

    if opt.log:
        log_train_file = os.path.join(opt.save_model_dir, 'train.log')
        log_valid_file = os.path.join(opt.save_model_dir, 'valid.log')
        log_test_file = os.path.join(opt.save_model_dir, 'test.log')

        print('[Info] Training performance will be written to file: {}, {}, and {}'.format(
            log_train_file, log_valid_file, log_test_file))

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf, open(log_test_file, 'a') as log_testf:
            log_tf.write('epoch,code_loss,code_accuracy,text_loss,text_accuracy\n')
            log_vf.write('epoch,code_loss,code_accuracy\n')
            log_testf.write('epoch,code_loss,code_accuracy\n')

    valid_code_accus = []
    for epoch_i in range(opt.resume_from_epoch, opt.resume_from_epoch + opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_code_loss, train_recon_loss, train_code_accu, train_recon_accu = train_epoch(
            modelTC, modelCT, training_data, mined_data, optimizer, device, smoothing=opt.label_smoothing, opt=opt, noise_model = noise_model)
        print('  - (Training)   code_loss: {code_loss: 8.5f}, code_accuracy: {code_accu:3.3f} %, recon_loss: {recon_loss: 8.5f}, recon_accuracy: {recon_accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  code_loss=train_code_loss, code_accu=100*train_code_accu, recon_loss=train_recon_loss, recon_accu=100*train_recon_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_code_loss, valid_code_accu= eval_epoch(
            modelTC, validation_data, device)
        print('  - (Validation)   code_loss: {code_loss: 8.5f}, code_accuracy: {code_accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  code_loss=valid_code_loss, code_accu=100*valid_code_accu,
                  elapse=(time.time()-start)/60))

        valid_code_accus += [valid_code_accu]

        start = time.time()
        test_code_loss, test_code_accu= eval_epoch(
            modelTC, test_data, device)
        print('  - (Test)   code_loss: {code_loss: 8.5f}, code_accuracy: {code_accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  code_loss=test_code_loss, code_accu=100*test_code_accu,
                  elapse=(time.time()-start)/60))

        if (epoch_i+1)%(opt.test_epoch) == 0:
            eval_bleu_score(opt, modelTC, validation_data, device, epoch_i, split = 'dev')

        if (epoch_i+1)%(opt.test_epoch) == 0:
            eval_bleu_score(opt, modelTC, test_data, device, epoch_i, split = 'test')

        modelTC_state_dict = modelTC.state_dict()
        modelCT_state_dict = modelCT.state_dict()
        checkpoint = {
            'modelTC': modelTC_state_dict,
            'modelCT': modelCT_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model_dir:
            if opt.save_mode == 'all':
                model_name = os.path.join(opt.save_model_dir, 'epoch_{0:02d}.chkpt'.format(epoch_i))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = os.path.join(opt.save_model_dir, 'epoch_{0:02d}.chkpt'.format(epoch_i))
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file and log_test_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf, open(log_test_file, 'a') as log_testf:
                log_tf.write('{epoch},{code_loss: 8.5f},{code_accu:3.3f},{recon_loss: 8.5f},{recon_accu:3.3f}\n'.format(
                    epoch=epoch_i, code_loss=train_code_loss, code_accu=100*train_code_accu, recon_loss=train_recon_loss, recon_accu=100*train_recon_accu))
                log_vf.write('{epoch},{code_loss: 8.5f},{code_accu:3.3f}\n'.format(
                    epoch=epoch_i, code_loss=valid_code_loss, code_accu=100*valid_code_accu))
                log_testf.write('{epoch},{code_loss: 8.5f},{code_accu:3.3f}\n'.format(
                    epoch=epoch_i, code_loss=test_code_loss, code_accu=100*test_code_accu))
                
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
    parser.add_argument('-use_noise', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    # For bleu eval
    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    
    parser.add_argument('-test_epoch', type=int, default=5, help='Test every x epochs')
    parser.add_argument('-resume_from_epoch', type=int, default=0, help='Warm restart')

    # New loss weighting
    parser.add_argument('-alpha', type=float,default=1.0, help='Weighting loss')
    parser.add_argument('-lr', type=float,default=1e-3, help='Learning rate')
    #parser.add_argument('-no_return_masks',dest = 'return_masks', default = True, action='store_false')
    #parser.add_argument('-no_return_logits',dest = 'return_logits', default = True, action='store_false')

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
    transformer = Transformer2(
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
            opt.tgt_vocab_size,
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
            betas=(0.9, 0.98), eps=1e-09, lr=opt.lr),
        opt.d_model, opt.n_warmup_steps)

    save_params(opt)

    opt = check_restart_conditions(opt)
    if opt.resume_from_epoch >= 1:
        print('Loading Old model')
        print('Loading model files from folder: %s' % opt.save_model_dir)
        transformer, transformer2 = load_models(transformer, transformer2, opt, opt.resume_from_epoch)

    if opt.use_noise:
    	noise_model = GaussianNoise()
    else:
    	noise_model = None

    train(transformer, transformer2, training_data, mined_data, validation_data, test_data, optimizer, device, opt, noise_model=noise_model)


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
            collate_fn=paired_collate_fn,
            shuffle=True)

    return train_loader, valid_loader, test_loader, mined_loader


if __name__ == '__main__':
    main()
