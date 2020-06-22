'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm
import os

from utils import normalization, renormalization, rounding, reverse_encoding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index


def gain (data_x, feature_name, onehotencoder, ori_data_dim, gain_parameters):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - feature_name: feature namelist of original data
    - onehotencoder: onehotencoder of this data
    - ori_data_dim: dimensions of original data    
    - gain_parameters: GAIN network parameters:
      - data_name: the file name of dataset
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      - onehot: the number of feature for onehot encoder (start from first feature)
      - predict: option for prediction mode
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  data_name = gain_parameters['data_name']
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  onehot = gain_parameters['onehot']
  predict = gain_parameters['predict']

  # Model Path
  model_path = 'model/'+ data_name
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector q 
  X = tf.placeholder(tf.float32, shape = [None, dim], name='X')
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim], name='M')
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim], name='H')
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim]), name='D_W1') # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]), name='D_b1')
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]), name='D_W2')
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]), name='D_b2')
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]), name='D_W3')
  D_b3 = tf.Variable(tf.zeros(shape = [dim]), name='D_b3')  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]), name='G_W1')  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]), name='G_b1')
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]), name='G_W2')
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]), name='G_b2')
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]), name='G_W3')
  G_b3 = tf.Variable(tf.zeros(shape = [dim]), name='G_b3')
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  saver = tf.train.Saver()
  if predict is True and os.path.exists(model_path + '.ckpt.meta'):
    print ("Model Restore")
    saver.restore(sess, model_path + '.ckpt')
  else:
    sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})
  if predict is False:
    save_path = saver.save(sess, model_path + '.ckpt')

  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)

  # Reverse encoding
  if onehot > 0:
    imputed_data = reverse_encoding(imputed_data, feature_name, onehotencoder, onehot, ori_data_dim)
          
  return imputed_data