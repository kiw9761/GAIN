'''Main function for GAIN.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import numpy as np

from data_loader import data_loader
from gain import gain

# Boolean input for stdin
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main (args):
  '''Main function
  
  Args:
    - data_name: the file name of dataset
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    - onehot: the number of feature for onehot encoder (start from first feature)
    - predict: option for prediction mode, no ramdom mask and save model if on
    
  Returns:
    - imputed_data_x: imputed data
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'data_name': args.data_name,
                     'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'onehot': args.onehot,
                     'predict': args.predict}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, feature_name, onehotencoder, ori_data_dim = data_loader(data_name, miss_rate, args.onehot, args.predict)
  
  # Impute missing data
  imputed_data_x = gain(miss_data_x, feature_name, onehotencoder, ori_data_dim, gain_parameters)

  # Save imputed data
  pd.DataFrame(imputed_data_x, columns= feature_name).to_csv ('data_imputed/' + data_name + "_imputed.csv", index = False, header=True)
  
  return imputed_data_x

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      default='new_data_051',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--onehot',
      help='number of onehot encoding columns',
      default=0,
      type=int)
  parser.add_argument(
      '--predict',
      help='Prediction Mode',
      default=False,
      type=str2bool)
  
  args = parser.parse_args()
  
  # Calls main function  
  imputed_data = main(args)
