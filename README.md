# MastersThesis
Masters thesis on complex word identification (CWI) for language learners

## Complex Word Identification: Sequential Model

## Dataset

The dataset used to train these models was collected by Yimam et al., (2018) and is available [here](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/complex-word-identification-dataset.html).

## Requirements
python (tested with 3.6)

tensorflow (tested with 1.3.0)

## Run with:

python experiment.py config.conf

experiment.py file is under CWI Sequence Labeller/sequence-labeler-master/

- download GloVe word embeddings 'glove.6B.300d.txt' from https://nlp.stanford.edu/data/glove.6B.zip and paste it under 'sequence-labeler-master/embeddings/glove' folder
- download FastText word embeddings 'crawl-300d-2M.vec' from https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip under 'sequence-labeler-master/embeddings/glove' folder
