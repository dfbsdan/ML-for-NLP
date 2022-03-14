# BERT Poolers
Implementation of different pooling methods for a sequence classification using a BERT Model.

The implemented poolers are:
- MeanMax.
- CLS (See src/transformers/models/bert modeling_bert.py#L610 at huggingface/transformers f43a425fe89cfc0e9b9aa7abd7dd44bcaccd79a) 
- A modified version of the stochastic pooling mechanism described by Hossein, G. and Hossein, K. (Hossein Gholamalinezhad and Hossein Khosravi. Pooling Methods in Deep Neural Networks, a Review. arXiv.org, 2020.). More information about this can be seen in the specs.pdf file.