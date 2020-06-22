# Originally Codebase for "Generative Adversarial Imputation Networks (GAIN)"

Fork from: Jinsung Yoon(jsyoon0823)

Authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

Paper: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GAIN: Missing Data Imputation using Generative Adversarial Nets," 
International Conference on Machine Learning (ICML), 2018.
 
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf

Place yours dataset in "data" folder.

To run the pipeline for training on GAIN framwork, simply run 
python3 -m main.py --data_name dataset.

Note that any model architecture can be used as the generator and 
discriminator model such as multi-layer perceptrons or CNNs. 

### Command inputs:

-   data_name: filename of dataset (don't include format)
-   miss_rate: probability of missing components
-   batch_size: batch size
-   hint_rate: hint rate
-   alpha: hyperparameter
-   iterations: iterations
-   onehot: number of feature for onehot encoder (start from first feature)
-   predict: option for prediction mode, no ramdom mask and save model if on


### Example command

```shell
$ python3 main.py --data_name dataset
--miss_rate 0.2 --batch_size 128 --hint_rate 0.9 --alpha 100
--iterations 10000 --onehot 5 --predict False
```

### Outputs

-   imputed_data_x: imputed data
