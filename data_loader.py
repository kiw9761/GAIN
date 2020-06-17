'''Data loader for GAIN.
'''

# Necessary packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils import binary_sampler



def data_loader (data_name, miss_rate, onehot):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: the filename of dataset
    - miss_rate: the probability of missing components
    - onehot: the number of feature for onehot encoder (start from first feature)
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
    feature_name: feature namelist of original data
    onehotencoder: onehotencoder of this data
    ori_data_dim: dimensions of original data
  '''
  
  # Load data
  file_name = 'data/'+data_name+'.csv'
  data = pd.read_csv(file_name)

  # Check dataset fit the train
  if data.shape[2] is not None:
    print("Dataset not 2D")

  # Onehotencoding, if columns have exist missing value, skip encoding
  onehotencoder = OneHotEncoder()
  if data[:,:onehot-1].isnull() == 0 and onehot > 0:
    data_x = data.iloc[:,:onehot-1]
    onehotencoder.fit(data_x)
    data_x = onehotencoder.transform(data_x).toarray()
    data_x = np.concatenate((data_x, data.iloc[:,5:].values),axis=1)
  elif onehot == 0:
    data_x = np.array(data)
  else:
    print("Missing value exist, skip onehotencoding")
    data_x = np.array(data)

  # Save feature name
  feature_name = list(data.columns)

  # Parameters
  ori_data_dim = data.shape[1]
  no, dim = data_x.shape
  
  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m, feature_name, onehotencoder, ori_data_dim