#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:35:50 2023

@author: eafpres
"""
#%% libraries
#
use_torch = True
#
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import gc
#
if not use_torch:
# suppress some annoying stuff
#
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
#
# tenorflow model and layers
#
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
  from tensorflow.keras.optimizers import Adam
#
# for tensorflow image utilities
#
  import tensorflow_addons as tfa
#
# fix for bug in CuDnn in Ubuntu
#
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#
# for dataset generator
#
  from tensorflow.keras.utils import image_dataset_from_directory
#
else:
  import torch
  torch.set_float32_matmul_precision('medium')
  import torch.nn as nn
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  from torchvision.datasets import ImageFolder
  from torchvision import transforms
  from torch.utils.data import DataLoader
  from torchmetrics.classification import BinaryAccuracy
#
#
#%% timestamp
#
def ts(ms = False, real = False):
  from datetime import datetime
  ts_str = str(datetime.now())
  timestamp = (ts_str[0:10] + '_' +
               ts_str[11:13] + '_' +
               ts_str[14:16] + '_' +
               ts_str[17:19])
  if ms:
    timestamp = (timestamp + '_' +
                 str(int(round(int(ts_str[20:30]) / 1000, 0))).zfill(3))
  if real:
    time = (int(timestamp[11:13]) * 60 * 60 * 1000 +
            int(timestamp[14:16]) * 60 * 1000 +
            int(timestamp[17:19]) * 1000)
    if ms:
      time = time + int(timestamp[20:])
    timestamp = time
  return timestamp
#
#%% instantiate a tensorflow dataset
#
def instantiate_dataset(model_path, 
                        data_path, 
                        directory,
                        batch_size,
                        color_mode = 'grayscale',
                        labels_from = 'inferred',
                        label_mode = 'categorical',
                        resize = 128,
                        random_seed = 42,
                        shuffle = False,
                        equalize = False,
                        normalize = True):
#
  root = \
    (data_path + directory)
  data = image_dataset_from_directory(
    root, 
    batch_size =  batch_size,
    shuffle = shuffle,
    labels = labels_from, 
    label_mode = label_mode,
    color_mode = color_mode,
    image_size = (resize, resize),
    seed = random_seed)
  if equalize: 
    if labels_from ==  None:
      h_equalize = lambda x: tfa.image.equalize(x) / 255
    else:
      h_equalize = lambda x, y: (tfa.image.equalize(x) / 255, y)
    data = data.map(h_equalize)
  elif normalize:
    if labels_from ==  None:
      normalize = lambda x: x / 255
    else:
      normalize = lambda x, y: (x / 255, y)
    data = data.map(normalize)
#    
  if labels_from ==  'inferred':
    data_files = \
      list(tf.data.Dataset.list_files(str(root + '/*/*.jpg'), 
                                      shuffle = shuffle,
                                      seed = random_seed))
  else:
    data_files = \
      list(tf.data.Dataset.list_files(str(root + '/*.jpg'), 
                                      shuffle = shuffle,
                                      seed = random_seed))
  data_files = \
    pd.DataFrame({'file' :
                  [data_files[i].numpy() 
                   for i in range(len(data_files))]},
                 index = range(len(data_files))).reset_index(drop = True)
#
  return data, data_files
#
#%% dataset configuration
#
def create_train():
  train, train_files = \
    instantiate_dataset(model_path = model_path,
                        data_path = data_path,
                        directory = train_dir,
                        batch_size = batch_size,
                        color_mode = 'grayscale',
                        labels_from = 'inferred',
                        label_mode = 'categorical',
                        resize = resize,
                        random_seed = random_seed,
                        shuffle = False,
                        equalize = False,
                        normalize = True)
  gc.collect()
  items = 0
  for index, item in enumerate(train):
    items +=  item[0].shape[0]
#
  return train, items, train_files
#
#%% configure
#
model_path = \
  ('/mnt/c/eaf_llc/aa-analytics_and_bi/alliance_molds/radome_quality/' +
   'image_analysis/models')
#
random_state = 42
#
data_path = \
  ('/mnt/c/eaf_llc/aa-analytics_and_bi/alliance_molds/radome_quality/' + 
   'image_analysis/data/processed/final_test')
#
# note that train_dir is required even if ''
#
train_dir = '/../20230211_blind_test/train'
train_folders = ['good', 'bad']
#
project = os.getcwd()
#
# resize controls the image scaling; too large and we run out of RAM
# limt data controls if we sample the images
# True & limit_fraction < 1 means that fraction is radomly sampled
#
resize = 128
greyscale = True
if greyscale:
  image_depth = 1
else:
  image_depth = 3
#
if not use_torch:
  epochs = 150
  learning_rate = 5e-4
  batch_size = 1024
else:
  epochs = 15
  learning_rate = 5e-4
  batch_size = 1024
random_seed = 42
#
#%% data for keras
#
start = ts(ms = True, real = True)
if not use_torch:
  train, train_size, train_files = create_train()
#
#%% configure for performance
#
if not use_torch:
  AUTOTUNE = tf.data.AUTOTUNE
  train = train.cache().prefetch(buffer_size = AUTOTUNE)
end = ts(ms = True, real = True)
print('time to load data for Keras ' + 
      str(round((end - start) / 1000, 0)) + 's')
#
#%% copy.ai code -- Keras
#
# Can you show me Python code using Keras for a basic CNN model?

# Sure! Here's an example of a basic CNN model using Keras in Python:

if not use_torch:
# Define the model
  model = Sequential()
  
# Add the first convolutional layer
  model.add(Conv2D(4, (3, 3), 
                   activation = 'relu', 
                   input_shape = (resize, resize, 1)))
  
# Add the pooling layer
  model.add(MaxPooling2D(pool_size = (2, 2)))
  
# Add the second convolutional layer
  model.add(Conv2D(16, (3, 3), activation = 'relu'))
  
# Add the second pooling layer
  model.add(MaxPooling2D(pool_size = (2, 2)))
  
# Flatten the output from convolutional layers
  model.add(Flatten())
  
# Add the first fully connected layer
  model.add(Dense(units = 128, activation = 'relu'))
  
# Add the output layer
  model.add(Dense(units = 2, activation = 'sigmoid'))
  
# Compile the model
  model.compile(optimizer = Adam(learning_rate = learning_rate),
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
  
# Train the model
  start = ts(ms = True, real = True)
  history = model.fit(train.shuffle(buffer_size = train_size,
                                    seed = random_seed,
                                    reshuffle_each_iteration = False), 
                      epochs = epochs, 
                      batch_size = batch_size)
  end = ts(ms = True, real = True)
  print('time to fit Keras ' + str(round((end - start)/ 1000, 0)) + 's')
#
# In this example, we've created a model with two convolutional layers, 
# each followed by a pooling layer, a fully connected layer with 128 units, 
# and an output layer with a sigmoid activation function. 
# We've compiled the model using the Adam optimizer 
# and binary cross-entropy loss, and specified accuracy 
# as the metric to monitor during training. 
# We've then trained the model
#
#%% evaluate Keras
#
train_labels = np.array([item 
                         for sublist in [np.argmax(labels.numpy(),
                                                   axis = 1).tolist() 
                                         for _, labels in train] 
                         for item in sublist])
train_pred = np.argmax(model.predict(train), axis = 1)
cm = confusion_matrix(train_labels, train_pred)
acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
#
#%% data for PyTorch
#
if use_torch:
  try:
    del train
    del train_files
  except:
    ...
#
  train_transform = transforms.Compose((transforms.Grayscale(),
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor()))
  root = \
    (data_path + train_dir)
  train = ImageFolder(root = root,
                      transform = train_transform)
  train_loader = DataLoader(train,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 16, 
                            pin_memory = True)
#
#%% copy.ai code -- PyTorch
#
# Can you show me the same model, but in PyTorch?

# Certainly! Here is an example of the same basic CNN model using PyTorch:

if use_torch:
  # Define the model
  class BasicCNN(nn.Module):
    def __init__(self):
      super(BasicCNN, self).__init__()
       
# Define the convolutional layers
      self.conv1 = nn.Conv2d(1, 4, kernel_size = 3, stride = 1, padding = 1)
      self.conv2 = nn.Conv2d(4, 16, kernel_size = 3, stride = 1, padding = 1)
  
# Define the pooling layers
      self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
  
# the layers 
      size = int(16 * (128 / 2 / 2)**2) # channels * (image size / # pooling / pooling factor)**2
      self.fc1 = nn.Linear(size, 128) 
      self.fc2 = nn.Linear(128, 1)
    
# Define the activation function
      self.relu = nn.ReLU()
     
 # Define the output activation function
      self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
    # Pass input through the convolutional layers
      x = self.conv1(x)
      x = self.relu(x)
      x = self.pool(x)
      x = self.conv2(x)
      x = self.relu(x)
      x = self.pool(x)
      
    # Flatten the output from convolutional layers
      size, _, _, _ = x.shape
      x = x.reshape(size, -1)
      
    # Pass input through the fully connected layers
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      
      return x
  
# Create the model instance
  model = BasicCNN()
  model = torch.compile(model)
  model.to(device)
  
# Define the loss function and optimizer
  criterion = nn.BCELoss()
  metric = BinaryAccuracy()
  metric.to(device)
  optimizer = torch.optim.Adam(model.parameters())
  
# Train the model
  start = ts(ms = True, real = True)
  for epoch in range(epochs):
    running_loss = 0
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
      count = labels.shape[0]
      inputs = inputs.to(device)
      labels = labels.to(device)
# Zero the gradients
      optimizer.zero_grad()
      
# Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, labels.float().unsqueeze(1))

# Backward pass
      loss.backward()
      optimizer.step()
      
# Print statistics
      running_loss +=  loss.item()
      running_acc = metric(outputs.flatten(), labels)
      if i % 20 == 0:
        print('@ ' + str(round((ts(ms = True,
                                   real = True) - 
                                start) / 1000, 0)) + 's' +
              ' epoch ' + str(epoch + 1) +
              ' step ' + str(i + 1) +
              ' acc ' + str(running_acc) +
              ' loss ' + str(running_loss / 100))
        running_loss = 0.0
  end = ts(ms = True, real = True)
#
  print('time to fit PyTorch ' + str(round((end - start) / 1000, 0)) + ' sec')
#
# In this example, we've defined a BasicCNN class that inherits 
# from nn.Module. We've defined the same convolutional layers, 
# pooling layers, fully connected layers, and activation
# functions as in the Keras example, but using PyTorch's building blocks.
# We've also defined the forward method, which specifies the order
# in which the inputs are passed through the layers.
#
# We've then created an instance of the BasicCNN class and 
# defined the loss function and optimizer. We've trained the model 
# using the Adam optimizer and binary cross-entropy loss,
# and printed the training loss every 100 batches.
#
#%% evaluate PyTorch
#
val_transform = transforms.Compose((transforms.Grayscale(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor()))
root = \
  (data_path + train_dir)
val = ImageFolder(root = root,
                  transform = val_transform)
val_loader = DataLoader(val,
                        batch_size = 4096,
                        shuffle = False,
                        num_workers = 16, 
                        pin_memory = True)
model.eval()
with torch.no_grad():
  preds = []
  ground_truth = []
  for (data, labels) in val_loader:
    preds = preds + (list(model(data.to(device))[:, 0].
                          to('cpu').detach().numpy()))
    ground_truth = ground_truth + list(labels.numpy())
#
all_preds = pd.DataFrame({'act' : ground_truth,
                          'raw_pred' : preds})
all_preds['pred'] = [0 if all_preds.raw_pred[i] < 0.5 else 1
                     for i in range(all_preds.shape[0])]
cm = confusion_matrix(all_preds.act, all_preds.pred)
acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
