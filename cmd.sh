python3 tct_train.py -data conala-trainnodev+mined-100k+test-16000.pt -snippet_model data/annotated/conala-trainnodev+mined-100k-snippet-16000.model -n_layers 2 -epoch 100 -batch_size 32 -beam_size 2 -dropout 0.2 -alpha 2.0 -save_model_dir /scratch/cluster/avrteja/nlp2/tct_train_mined_check -test_epoch 1
