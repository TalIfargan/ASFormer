# ASFormer Experiments
This repo is based on the github repo from the BMVC 2021 paper: [ASFormer: Transformer for Action Segmentation](https://arxiv.org/pdf/2110.08568.pdf).

## Main Components
`main.py` - for running an experiment.

`model.py` - includes most of the code for model building and training.

`train_efficient.py` - code used to train Efficientnet as a part of the experiments described in our work.

Also, this repo contains all code needed for running the experiments described in our work, such as a new label defenition for the  "transition classes", dimension reduction for the input features and more.

## Results
In addition to the results extensively described in our work, this repo contatins all the outputs of our experiments.

In any results directory, the results for the relevant test-set is provided. the results for each video contain its raw gesture-recognition output, and 3 segmentation images describing the outputs of the ASFormer (the third one is the one taken as the model's output). In any segmentation image we can a graph of the model's certainty along the frames.
