# Action Segmentation for Surgical Data
Please see project description, goals, methods and results in the final report [Report](Project_report.pdf)

NOTE:
This repo is based on the github repo from the BMVC 2021 paper: [ASFormer: Transformer for Action Segmentation](https://arxiv.org/pdf/2110.08568.pdf).

## Main Components
`main.py` - for running an experiment.

`model.py` - includes most of the code for model building and training.

`train_efficient.py` - code used to train Efficientnet as a part of the experiments described in our work.

Also, this repo contains all code needed for running the experiments described in our work, such as a new label defenition for the  "transition classes", dimension reduction for the input features and more.

## Results
Output videos can be downloaded from: https://drive.google.com/drive/folders/1S_tcUdrOZF3vKk1Ow9YVfAd2Jv9wyxcD?usp=sharing 
In addition to the results extensively described in our work, this repo contatins all the outputs of our experiments.

In any results directory, the results for the relevant test-set is provided. the results for each video contain its raw gesture-recognition output, and 3 segmentation images describing the outputs of the ASFormer (the third one is the one taken as the model's output). In any segmentation image we can also see a graph of the model's certainty along the frames.

## Reproduction
### First Step: Training EfficientNet With Transition Classes
First, run `data_modifier.py` in order to save the new defined labels for the feature extractor.

Then, use the command `python train_efficient.py --FOLD {fold_num}`, where `fold_num` can be from 0 to 4.

### Second Step: Feature Making - Saving the Inference of Efficientnet
`python feature_maker.py`

### Third step: Dimension Reduction
`python pca_maker.py --output_size {size} --split {split_num}`, where `size` is the desired features dimension and `split_num` is the fold number.

### Fourth step: ASFormer Training
`python main.py --split {split_num} --hidden {hidden_dim}`, where split_num is the fold number and hidden_dim is the desired feature-space size to use.

### Fifth step: ASFormer Evaluation
`python main.py --split {split_num} --hidden {hidden_dim} --action predict`

### Extra Step: Video Making
`python video_maker.py`, will run on 3 previously picked videos based on the extracted results.
