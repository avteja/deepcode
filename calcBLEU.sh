#!/bin/bash
python preproc/seq2seq_output_to_code.py results/mypreds.hyp data/conala-corpus/conala-$1.json.seq2seq results/$1.json
python eval/conala_eval.py --strip_ref_metadata --input_ref data/conala-corpus/conala-$1.json --input_hyp results/$1.json --output_dir $2 --epoch $3
