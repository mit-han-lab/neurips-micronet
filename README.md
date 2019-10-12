# MicroNet: MuLan

## Introduction
This codebase is our submission for MicroNet Challenge on the WikiText-103 Language Modeling task.

The codebase consists of:
* A IPython notebook which verifies test perplexity and walk through the calculations (storage and FLOP): [LINK]
* A write-up which describes our methodology and calculations in details: [LINK]
* Three model paths of our submissions: [LINK1] [LINK2] [LINK3]


## Method 
We use an efficient primitive modified based on Transformer-XL. Specifically, we make use of shorter attention length, trainable cache, Hebian updates, and a tied adaptive embedding and softmax layer.
After training our primitive model, we run knowlege distillation to get a better performance. Finally, we run prunning and quantization to futher compress the model.
For more details, see our write-up: [LINK]

## Submission 

ID, score, sparsity, quantization bit, test perplexity, ipython
0.0482 / 
0.0485
0.0475 / 8 bit / 

