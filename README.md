# BioCreativeVI-PM-Track Document Triage Task
***
## Dependency package

BioCreativeVI-PM-Track Document Triage Task uses the following dependencies:

- [Python 2.7](https://www.python.org/)
- [keras 2.0.9](https://keras.io/)
- [numpy 1.12.1](http://www.numpy.org/)


## Content
- data
	- PPIm: the BioCreativeVI PM Track Document Traige corpus
	- PPI: the previous BioCreative PPI corpora
- src
	- Represent_luo.py
	- Eval.py
	- FileUtil.py
	- Load_dataset.py
	- PreProcessing.py
	- AttentionLayer.py
	- PPIAC-LSTM-pretrain.py: pre-train a PPI model
	- Hie_RNN.py: train a HieLSTM model
	- Hie_RNN-Classifier.py: classify the document using the HieLSTM model
	- PPIm-Feature-ppiac.py: train a PPIm model (including the LSTM, CNN, LSTM-CNN and RCNN models)
	- PPIm-Feature-ppiac-classifier.py: classify the document using the PPIm model (including the LSTM, CNN, LSTM-CNN and RCNN models)
	- Ensemble-LR.py: the ensemble using a LR model
	- Ensemble-voting.py: the ensemble using the voting
- fea_vocab
	- POS.vocab: the lookup table of the POS feature
	- NER.vocab: the lookup table of the NER feature


## Models

The trained models can be downloaded from [https://www.kaggle.com/lingluodlut/biocreativevipmtrackmodels/data](https://www.kaggle.com/lingluodlut/biocreativevipmtrackmodels/data).

- models
	- BioCreativevi.rar: the 50-dimensional word embedding
	- bilstm-att-token-1l-50d-ppipre.rar: the pre-trained PPI model
	- models-ppipre-nofea.rar: the models without additional features (including LSTM, CNN, LSTM-CNN, RCNN, HieLSTM and a ensemble model using a logistic regression)
