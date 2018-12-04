''' Handling the data io '''
import argparse
import json
import random
import torch
import transformer.Constants as Constants
import sentencepiece as spm

def read_instances_from_file(sp, inst_file, word2idx):
    ''' Convert file into word seq lists and vocab '''
    bos_index = word2idx[Constants.BOS_WORD]
    eos_index = word2idx[Constants.EOS_WORD]

    token_insts = []
    max_len = 0
    lens = []
    with open(inst_file) as f:
        for sent in f:
            sent = sent.strip()
            ids = [bos_index] + sp.EncodeAsIds(sent) + [eos_index]
            # if 0 in ids:
            #     print ('unk present')
            #     print (sent)
            #     print (ids)
            # pieces = sp.EncodeAsPieces(sent)
            token_insts += [ids]
            max_len = max(max_len, len(ids))
            lens.append(len(ids))

    lens.sort()
    # print (lens)
    print('[Info] Get {} examples from {} with max length {}'.format(len(token_insts), inst_file, max_len))

    return token_insts

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_intent')
    parser.add_argument('-train_snippet')
    parser.add_argument('-valid_intent')
    parser.add_argument('-valid_snippet')
    parser.add_argument('-test_intent')
    parser.add_argument('-test_snippet')
    parser.add_argument('-intent_model')
    parser.add_argument('-snippet_model')
    parser.add_argument('-intent_vocab')
    parser.add_argument('-snippet_vocab')
    parser.add_argument('-save_data')
    parser.add_argument('-max_len', type=int, default=200)

    opt = parser.parse_args()

    sp_intent = spm.SentencePieceProcessor()
    sp_intent.Load(opt.intent_model)
    sp_snippet = spm.SentencePieceProcessor()
    sp_snippet.Load(opt.snippet_model)

    intent_idx2word = {}
    intent_word2idx = {}
    with open(opt.intent_vocab) as f:
        i = 0
        for row in f:
            r = row.strip().split()
            intent_idx2word[i] = r[0]
            intent_word2idx[r[0]] = i
            i += 1
        i = Constants.PAD
        intent_idx2word[i] = Constants.PAD_WORD
        intent_word2idx[Constants.PAD_WORD] = i

    snippet_idx2word = {}
    snippet_word2idx = {}
    with open(opt.snippet_vocab) as f:
        i = 0
        for row in f:
            r = row.strip().split()
            snippet_idx2word[i] = r[0]
            snippet_word2idx[r[0]] = i
            i += 1
        i = Constants.PAD
        snippet_idx2word[i] = Constants.PAD_WORD
        snippet_word2idx[Constants.PAD_WORD] = i

    # Training set
    train_intent_insts = read_instances_from_file(sp_intent, opt.train_intent, intent_word2idx)
    valid_intent_insts = read_instances_from_file(sp_intent, opt.valid_intent, intent_word2idx)
    test_intent_insts = read_instances_from_file(sp_intent, opt.test_intent, intent_word2idx)

    train_snippet_insts = read_instances_from_file(sp_snippet, opt.train_snippet, snippet_word2idx)
    valid_snippet_insts = read_instances_from_file(sp_snippet, opt.valid_snippet, snippet_word2idx)
    test_snippet_insts = read_instances_from_file(sp_snippet, opt.test_snippet, snippet_word2idx)

    indices = [idx for idx in range(len(train_snippet_insts)) if len(train_snippet_insts[idx]) <= opt.max_len ]
    train_intent_insts = [train_intent_insts[i] for i in indices]
    train_snippet_insts = [train_snippet_insts[i] for i in indices]

    opt.train_max_input_len = max([len(intent) for intent in train_intent_insts])
    opt.train_max_output_len = max([len(snippet) for snippet in train_snippet_insts])
    opt.valid_max_input_len = max([len(intent) for intent in valid_intent_insts])
    opt.valid_max_output_len = max([len(snippet) for snippet in valid_snippet_insts])
    opt.test_max_input_len = max([len(intent) for intent in test_intent_insts])
    opt.test_max_output_len = max([len(snippet) for snippet in test_snippet_insts])

    print ('Number of Train Instances =', len(train_intent_insts))
    print ('Train Max Input Len =', opt.train_max_input_len)
    print ('Train Max Output Len =', opt.train_max_output_len)

    assert len(train_intent_insts) == len(train_snippet_insts), '[Error] The training instance count is not equal.'
    assert len(valid_intent_insts) == len(valid_snippet_insts), '[Error] The valid instance count is not equal.'
    assert len(test_intent_insts) == len(test_snippet_insts), '[Error] The test instance count is not equal.'

    data = {
        'settings': opt,
        'dict': {
            'src': intent_word2idx,
            'tgt': snippet_word2idx},
        'train': {
            'src': train_intent_insts,
            'tgt': train_snippet_insts},
        'valid': {
            'src': valid_intent_insts,
            'tgt': valid_snippet_insts},
        'test': {
            'src': test_intent_insts,
            'tgt': test_snippet_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
