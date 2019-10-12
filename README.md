# MicroNet: MuLan

## Introduction
This codebase is our submission for MicroNet Challenge on the WikiText-103 Language Modeling task.

The codebase consists of:
* IPython notebooks which verify test perplexity and calculate the final score
* A write-up which describes our methodology and calculations in details
* Three model paths of our submissions (in WikiText-103)

*We run our codebase on Python 3.6.8 and Pytorch 1.1.0*

## Method 
We use an efficient primitive modified based on Transformer-XL. Specifically, we make use of shorter attention length, trainable cache, Hebian updates, and a tied adaptive embedding and softmax layer.
After training our primitive model, we run knowlege distillation to get a better performance. Finally, we run prunning and quantization to futher compress the model.
For more details, see our write-up: [report](report.pdf)

## Submission 

| ID  | Notebook  | Sparsity (%) | Quantization (Bit)| Test PPL | **Score** |
| --- |:---------:| --------:|-------------:|---------:|----------:|
| 1 | [Notebook1](micronet_challenge-wikitext_103-1.ipynb)| 42.12 | 9 | 34.95 | **0.0482** |
| 2 | [Notebook2](micronet_challenge-wikitext_103-2.ipynb)| 40.12 | 9 | 34.65 | **0.0485** |
| 3 | [Notebook3](micronet_challenge-wikitext_103-3.ipynb)| 33.85 | 8 | 34.95 | **0.0475** |

Our best submission has score **0.0475**.

## Verification
To verify test perplexity and score calculation, please refer to the corresponding IPython Notebooks. 
Before running the notebooks, it's required to download the data ("cache/" folder) at: [data](link) 


## FAQ
If you have any further questions about our submission, please don't hesitate reaching out to us through Github Issues :)
Thanks!

