If you want to use allennlp elmo in your pytorch code it can still be a nightmare. 


`batch_to_ids` function actually call 

`allennlp.data.token_indexers.elmo_indexer.ELMoTokenCharactersIndexer` 
aka `elmo_characters` in dataset_reader


`elmo_token_embedder` is actually doing the embedding


Can also see 

https://github.com/allenai/allennlp/blob/master/tutorials/how_to/training_transformer_elmo.md

https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L27




Weighted elmo
https://github.com/allenai/allennlp/issues/2397
>>> salar = [0.5, 0.3, 0.2]
>>> elmo = Elmo(options_file, weight_file, salar, 1, dropout=0)


