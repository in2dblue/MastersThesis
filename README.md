# MastersThesis
Masters thesis on complex word identification (CWI) for language learners

# Complex Word Identification

This github contains the state-of-the-art complex word identification model *SEQ* from [Gooding and Kochmar (2019)](https://www.aclweb.org/anthology/P19-1109). 

# Dataset

The dataset used to train these models was collected by Yimam et al., (2018) and is available [here](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/complex-word-identification-dataset.html).

# System Output

The system output for both *SEQ* and the winning shared task submission *CAMB* is available in [Sytem Output](./System%20Output).

## Requirements
python (tested with 3.6)

tensorflow (tested with 1.3.0 and 1.4.1)

## Run with:

python experiment.py config.conf

experiment.py file is under CWI Sequence Labeller/sequence-labeler-master/

- download file 'glove.6B.300d.txt' from https://nlp.stanford.edu/data/glove.6B.zip and paste it under 'sequence-labeler-master/embeddings/glove' folder
