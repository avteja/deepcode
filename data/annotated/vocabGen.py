import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=conala-trainnodev.intent --model_prefix=conala-trainnodev-intent-4000 --vocab_size=4000 --max_sentence_length=10000 --hard_vocab_limit=False')
spm.SentencePieceTrainer.Train('--input=conala-trainnodev.snippet --model_prefix=conala-trainnodev-snippet-4000 --vocab_size=4000 --max_sentence_length=10000 --hard_vocab_limit=False')
