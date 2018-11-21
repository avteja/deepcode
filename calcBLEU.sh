#!/bin/bash
python preproc/seq2seq_output_to_code.py results/mypreds.hyp data/conala-corpus/conala-dev.json.seq2seq results/dev.json
python eval/conala_eval.py --strip_ref_metadata --input_ref data/conala-corpus/conala-dev.json --input_hyp results/dev.json