# Reserch-Papers
## Paper-1: Hateful Memes Classification using Machine Learning : Jafar Badour, Joseph Alexander Brown
### Problem statement
Classification problem of multimodal data dedicated to classifying hateful memes using two approaches i.e. tate-of-the-art image and textual
feature extractors
### Methodology / Solution / Architecture
Approaches:-
1. converting the visual channel into a textual one and feed it to textual classifiers
2. converted both channels into a vector representation and then combined them to represent the visual-textual context
####    Visual attention and Textual Classifiers Using LSTM.
-  First the meme is fed into the image captioning model then the caption and the meme text are concatenated and fed into a two layered bidirectional LSTMs which finally is fed into a fully connected layer with a sigmoid activation
####    Object detection and MultiModal Classifiers using SBERT and Segmentation & Xception Model.
- each meme’s image is fed into a segmentation model and then the biggest three objects are fed into an Xception model for feature extraction. At the same time the textual channel of the meme is fed into Sentence Bert model and then all features from visual and textual channels are concatenated and fed into a fully connected Neural network
### Metrics
- The primary metrics used are Accuracy and Area Under the Curve (AUC).
### Experiment
- Visual Attention + Textual Classifier model
-  Sbert + Multimodal Classification Model.
### Code repository / supplemental material

- https://ieeexplore.ieee.org/abstract/document/9659896 
- https://github.com/JafarBadour/Hateful-Memes-Classification
### Result
| Model                               | Dataset | Acc  | Auc |
|-------------------------------------|:----------:|-----|------|
|Top Solution in FB challenge| Fb| 0.89| 0.8827|
|SBERT+ Multimodal Classifier | Fb| 0.75 |0.794|
|Visual Attention + Textual Classifier |Fb |0.642| 0.648
|Top Solution in FB challenge| Innopolis| 0.735 |0.8209|
|SBERT+ Multimodal Classifier | Innopolis |0.722| 0.8|
|Visual Attention + Textual Classifier | Innopolis |0.648 |0.64|

The percentage of dataset for training is 80% for all approaches for each dataset

### Conclusion
- Introduces a new dataset  consists of well over 20,000 memes, with 13% of them being hateful.
- Using segmentation to split the image into multiple images with lower complexity produced superior results over-generalized image captioning and textual classifiers
- Provides a free software solution that helps to annotate images, as well as, software to collect images from the web efficiently.

### Future work
- Adding more samples labelled by more annotators in Innopolis Hateful Memes dataset
-  Sentence BERT can be replaced by another model that encodes texts and provides a vectorized text representation
-  The multimodal classifier can take more objects and concatenate their Xception representation before feeding them into fully connected layers



## Paper-2: The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes :Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, Davide Testuggine
### Problem statement
Proposing a new challenge set for multimodal classification focusing on detecting hate speech in multimodal memes
### Methodology / Solution / Architecture
#### Steps
Reconstructing Memes using Getty Images
##### Annotation Process
- Phase 1: Filtering
- Phase 2: Meme construction
- Phase 3: Hatefulness rating
- Phase 4: Benign confounders
##### Splits 
-------------------
#### Models from three classes were evaluated:-
1. unimodal models,
2. multimodal models (unimodally pretrained)
3.  multimodal models(multimodally pretrained)

### Metrics
- Accuracy and Area Under the Curve (AUC).
### Experiment
-  MMBT, ViLBERT,  Visual BERT, text BERT, ViLBERT CC, Visual BERTCOCO
### Code repository / supplemental material
- https://proceedings.neurips.cc/paper/2020/file/1b84c4cee2b8b3d823b30e2d604b1878-Paper.pdf
- https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes

### Result

|Type                               | Model       | Validation | Validation| Test | Test |
|-----------------------------------|:-----------:|:----------:|:---------:|:--------:|:--------|
|                                    |            | Acc.       | AUROC     | Acc. | AUROC|
|                                   |Human  |   -  |   - | 84.70| - |
| Unimodal |Image-Grid| 50.67 |52.33 |52.73±0.72 |53.71±2.04 |
| Unimodal|Image-Region | 52.53 | 57.24 | 52.36±0.23 | 57.74±0.73 |
| Unimodal|Text BERT| 58.27 | 65.05|  62.80±1.42 | 69.00±0.11| 
|Multimodal (Unimodal Pretraining) |Late Fusion| 59.39| 65.07 |63.20±1.09| 69.30±0.33|
|Multimodal (Unimodal Pretraining)|Concat BERT| 59.32| 65.88| 61.53±0.96 | 67.77±0.87
|Multimodal (Unimodal Pretraining)|MMBT-Grid |59.59 | 66.73| 62.83±2.04 |69.49±0.59
|Multimodal (Unimodal Pretraining)|MMBT-Region| 64.75 | 72.62 | 67.66±1.39 | 73.82±0.20
|Multimodal (Unimodal Pretraining)|ViLBERT| 63.16 | 72.17 | 65.27±2.40| 73.32±1.09
|Multimodal (Unimodal Pretraining)|Visual BERT |65.01 |74.14 |66.67±1.68| 74.42±1.34
|Multimodal (Multimodal Pretraining)|ViLBERT CC| 66.10 |73.02 |65.90±1.20 |74.52±0.06|
|Multimodal (Multimodal Pretraining)|Visual BERTCOCO |65.93 |74.14| 69.47±2.06 | 75.44±1.86   |

They had a different set of annotators label the final test set, and obtained a human accuracy of 84.7% with low variance (tenbatches were annotated by separate annotators, per-batch accuracy was always in the 83-87% range).
### Conclusion
- Introduced a new challenge dataset and benchmark centered around detecting hate speech in multimodal memes
- Results on the task reflected a concrete hierarchy in multimodal sophistication, with more advanced fusion models performing better
### Future work
-  This challenge is meant to spur innovation and encourage new developments in multimodal reasoning and understanding, which can have positive effects for an extremely wide variety of tasks and applications.


## Paper-3 : Memeify: A Large-Scale Meme Generation System :Suryatej Reddy Vyalla∗, Vishaal Udandarao∗, Tanmoy Chakraborty
### Problem statement
Meme datasets available online are either specific to a context or contain no class information, proposed Memeify (meme generation system) which can creatively generate captions given a context. 
### Methodology / Solution / Architecture
- They  infered that the theme of a meme depends on the words in its caption and not the background image.
#### Paper is based on two major folds: 
1. generation of a large-scale meme dataset, which to our knowledge,
2. design of a novel web application to generate memes in real time.
##### Data Curation
- Created a dataset of 1.1 million memes belonging to 128 classes By data  web scrapped from QuickMeme and a few other sources
- Create an average word embedding of the captions to perform clustering
- Segregate meme classes into themes as identified by the clustering algorithm and labeled them as “Savage", “Depressing",“Unexpected", “Frustrated" , “Wholesome" and  “Normie"
----------------------------
#### Architecture
##### Generation Model
- Used the transformer based GPT-2 architecture  as  base language generative model.
- Used generative model trained with the class information as the caption generator in the Memeify system which enables the generation of memes in two specific ways:
-  (1) Randomization
-  (2) Customization
##### Web Application
- developed a web interface for users to interact with our meme generation algorithm

### Metrics
-  precision, recall, accuracy and F1-score
### Experiment
- Transformer based GPT-2 architecture, pretrained VGG16 convnet, AJAX

### Code repository / supplemental material
- https://arxiv.org/abs/1910.12279
- https://github.com/suryatejreddy/Memeify

### Result

|Metric      |Baseline       | Our model | 
|-------------|:-----------:|:----------:|
| Precision |64.28 |56.52|
| Recall |90 |86.66|
| Accuracy| 70 |60|
|F1-score |75.0 |68.42|

Theme Recovery
|Theme      |Accuracy  | 
|-----------|:------:|
|Normie |77.3|
|Savage| 86.1|
|Depressing |84.6|
|Unexpected |90.2|
|Frustrated| 87.7|
|Wholesome| 86.8|
|Overall |85.5|

Overall accuracy and per-theme accuracy for the classification


### Conclusion
- Exlpained Memeify which  is capable of generating memes either from existing classesand themes or from custom images.
- created a large-scale meme dataset consisting of meme captions, classes and themes
### Future work
They are interested in extending the Memeify system to include multiple parts in the meme caption


