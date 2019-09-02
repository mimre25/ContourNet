# ContourNet

This repository contains the PyTorch implementation for _ContourNet_, a Convolutional Neural Network (CNN) to detect salient isocontours in XGC plasma fusion.

# Dependencies

There is a conda environment (contournet.yml), however the basic requirements a pytorch, numpy, and scikit learn, as well as tqdm for progress bars.

Tested with   
- pytorch 1.1.0
- scikit-learn 0.21.2
- tqdm 4.32.1
- numpy 1.16.4

# Running

To run the training, simply call ``python Training.py``. 
For Inference, you need to sepcify the model name ``python Inference.py <model>`` that has been saved by Training.

# Citation

## todo
