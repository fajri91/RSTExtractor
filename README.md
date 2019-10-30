# RSTExtractor

This code is the combination of:
1. NeuroNLP2 (https://github.com/XuezheMax/NeuroNLP2) -- paper: Deep Biaffine Attention for Neural Dependency Parsing (https://arxiv.org/abs/1611.01734).
2. Neural RST Parser (Our implementation in Pytorch) -- paper: Transition-based Neural RST Parsing with Implicit Syntax Features (https://www.aclweb.org/anthology/C18-1047/).

This code is used to extract:
1. Latent feature of discourse units.
2. Shallow feature of discourse units.

For more technical details, please refer to our paper: Fajri Koto, Jey Han Lau, Timothy Baldwin. _Improved Document Modelling with a Neural Discourse Parser_.  In Proceedings of the 2019 Australasian Language Technology Workshop, Sydney. (to appear).

## Dependencies and Installation
1. Python 2.7
2. Run `pip install -r requirements.txt`

## Pre-Extraction
There are three main steps:
1. Using standford corenlp. After downloading the appropriate stanford corenlp, please run `python corenlp.py --source=PATH_TO_YOUR_DOCUMENTS/* --target=PATH_TO_YOUR_OUTPUT`.  Please make sure you put all the necessary files of stanford corenlp in this repo with a folder name `stanford-corenlp`.
2. For the next two steps, please follow https://github.com/fajri91/DPLP for:
  * Converting XML file to CoNLL format.
  * Segmenting CoNLL file to get EDUs. The output is *.segment file.

## Extraction
Now you are ready to extaract latent/shallow features as well as the RST tree.
1. For latent feature, please run `python extract_latent_feature.py`
2. For shallow feature, please first run `python extract_tree.py` and after that run `python extract_shallow_feature.py`

Note1: Please manually adjust all PATHs in the code as I have'nt implemented args.parse in the code.
Note2: Our RST parser performance is similar with Transition-based Neural RST Parsing with Implicit Syntax Features (https://www.aclweb.org/anthology/C18-1047/).
