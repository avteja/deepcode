python preprocess_conala.py -train_intent data/annotated/conala-trainnodev.intent -valid_intent data/annotated/conala-dev.intent -train_snippet data/annotated/conala-trainnodev.snippet -valid_snippet data/annotated/conala-dev.snippet -intent_model data/annotated/conala-trainnodev-intent-4000.model -snippet_model data/annotated/conala-trainnodev-snippet-4000.model -intent_vocab data/annotated/conala-trainnodev-intent-4000.vocab -snippet_vocab data/annotated/conala-trainnodev-snippet-4000.vocab -save_data conala-all.pt