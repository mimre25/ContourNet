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
For Inference, you need to specify the model name ``python Inference.py <model>`` that has been saved by Training.
The label propagation can either be run in a non-sequential manner (worse) or a sequential manner. 
The program parameters are the labeled time step, the slice and then a list of time step that should be propagated from the host time step.
For the sequential version, several calls have to be made, e.g. ``python LabelPropagation.py 60 10 61``, ``python LabelPropagation.py 61 10 62``, and so on. 
In this case, the propagation would first go from 60 to 61, and then from 61 to 62.
The non-sequential call in the following example ``python LabelPropagation.py 60 10 61 62`` would first propagate from 60 to 61 and then from 60 to 62, which leads to a worse output quality. 

# Citation

> Martin Imre, Jun Han, Julien Dominski, Michael Churchill, Ralph Kube, Choong-Seock Chang, Tom Peterka, Hanqi Guo, and Chaoli Wang. ContourNet: Salient Local Contour Identification for Blob Detection in Plasma Fusion Simulation Data. In Proceedings of International Symposium on Visual Computing, Lake Tahoe, NV, Oct 2019.
