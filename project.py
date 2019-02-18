
# coding: utf-8

# # Neural Composer project
# -----
# 

# In[1]:


import sys
sys.path.append("helpers/")

#Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

print "torchversion: ", torch.__version__

#Custom libraries
import dataset
import datapreparation
from datapreparation import plot_loss
from train import train
from models import Generalist, Specialist

#PATHS
DATASET_PATH = 'datasets/training/'
FS1 = 'piano_roll_fs1/'
FS2 = 'piano_roll_fs2/'
FS5 = 'piano_roll_fs5/'
FULL_PATH = DATASET_PATH + FS5


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# # Inspect the dataset

# In[2]:



#LOAD
dataset_names = datapreparation.load_all_dataset_names(FULL_PATH)
print(dataset_names)
datasets = datapreparation.load_all_dataset(FULL_PATH)
#datasets_df = pd.DataFrame(datasets)

#CONSTANTS
max_length = datapreparation.get_max_length(datasets)
num_keys = datapreparation.get_numkeys(datasets)[0]
print("max_length: ", max_length)
print("num_keys: ", num_keys)


# # Split datasets

# In[3]:


# shuffles the dataset on creation
Dataset = dataset.pianoroll_dataset_batch(FULL_PATH)
# splits the datasets
training_data, validation_data, testing_data = Dataset.split_datasets(split=[0.7,0.85, 1.0])


# # Sample from dataset

# In[4]:


#VISUALIZE
print('Bach')
print('timesteps in song:', len(datasets[0][0]))
datapreparation.visualize_piano_roll(datasets[0])

datapreparation.embed_play_v1(datasets[0])



# ----
# # Generalist composer
# The aim is to train a neural network that can create music in general. It shall be kickstarted by real music. Meaning that it will get some seconds of real music and then it wil continue playing music.

# In[5]:


# set parameters for the model
input_size = num_keys
hidden_size = 64
num_tags = len(pd.unique(dataset_names))
num_layers = 1

# create model
generalist = Generalist(input_size, hidden_size, num_layers)

# set parameters for the training process
name = "generalist_BCE_1e-2"
epochs = 1000
batch_size = 4 # does not effect training. need to implement in def train
lr = 1e-2
criterion = None # it will create default criterion
optimizer = None # it will create default optimizer


# In[ ]:


# TRAIN GENERALIST

trained_generalist, gen_train_loss, gen_val_loss, gen_train_acc, gen_test_acc  = train(generalist, training_data, validation_data, name, criterion, epochs, lr, optimizer, batch_size)


# In[6]:


# load loss
loaded_generalist_metrics = pickle.load(open( 'saved_losses/generalist_BCE_1e-2_999.pickle', "rb" ))
plot_loss(loaded_generalist_metrics)


# ### Generating songs

# In[7]:


# generate song from generalist

# pick song
#song = training_data[10]
song = testing_data[1]
# load best model
model = Generalist(input_size, hidden_size, num_layers) # need to know input_size, hidden_size excetera.. :/
model.load_state_dict(torch.load('saved_models/generalist_BCE_1e-2_208.pth')) #best so far on song test:1, train:10 with init=10 not good now
#model.load_state_dict(torch.load('saved_models/generalist_BCE_1e-2_175.pth'))


# In[8]:


# generate cherrypicked overfitted Specialist
print "Bach=0, Brahms=1, debussy=2, mozart=3"
print "composer number:", song[1].item()
datapreparation.gen_music_seconds(model,init=song[0],composer=song[1],fs=5,gen_seconds=60,init_seconds=20)


# ----
# # Specialist composer
# The aim is to train a neural network that can create music similar to a specific composer. It shall be kickstarted by a song from that composer. Meaning that it will get some seconds of the song and then it should be able to compose music that is typically similar to that specific composer

# In[9]:


# set parameters for the model
input_size = num_keys
hidden_size = 64
num_tags = len(pd.unique(dataset_names))
num_layers = 1

# create model
specialist = Specialist(input_size, hidden_size, num_tags, num_layers)

# set parameters for the training process
name = "specialist_BCE_1e-2"
epochs = 1000
batch_size = 4 # does not effect training. need to implement in def train
lr = 1e-2
criterion = None # it will create default criterion
optimizer = None # it will create default optimizer


# In[ ]:


# TRAIN SPECIALIST

trained_specialist, spe_train_loss, spe_val_loss, spe_train_acc, spe_test_acc  = train(specialist, training_data, validation_data, name, criterion, epochs, lr, optimizer, batch_size)


# In[10]:


# load loss
loaded_specialist_metrics = pickle.load(open( 'saved_losses/specialist_BCE_1e-2_999.pickle', "rb" ))
plot_loss(loaded_specialist_metrics)


# ### Generating songs

# In[11]:


# generate song from specialist

# pick song
#song = training_data[10]
song = testing_data[3]

# load best model
model = Specialist(input_size, hidden_size, num_tags, num_layers) # need to know input_size, hidden_size excetera.. :/
model.load_state_dict(torch.load('saved_models/specialist_BCE_1e-2_999.pth')) #68 is "best", but 999 is much better than 68 especially good on test[3]
# works okay with one second as well


# In[12]:


# generate cherrypicked overfitted Specialist
print "Bach=0, Brahms=1, debussy=2, mozart=3"
print "composer number:", song[1].item()
datapreparation.gen_music_seconds(model,init=song[0],composer=song[1],fs=5,gen_seconds=60,init_seconds=10)


# # Tested
# - Two GRU layers
# - MSEloss
# 
# 
# # Further works
# issue 1: - specializes on one song because of small sample size with huge samples
# - bigger dataset which there is here: http://www.piano-midi.de/
# - chunks should maybe remove some of that effect
# 
# # Ideas
# - implement teacher-forcing, but not sure how...
# - implement scheduler to decrease learningrate after 150 epochs
# - put some engenieering on the "prediction" of notes (penalizing holding tone for too long, )
# - Make CategoricalEntropy work
# - try to properly normalize prediction
# 
# 
# 

# # Test-case with new song

# In[13]:


repo_folder_path = "/Users/havardbjornoy/GitHub/music-generation"
PATH_TO_FILE = FULL_PATH #put in repo_folder_path + 'piano_roll_fs5/'

# shuffles the dataset on creation
Dataset = dataset.pianoroll_dataset_batch(FULL_PATH)
# splits the datasets
training_data, validation_data, testing_data = Dataset.split_datasets(split=[0.7,0.85, 1.0])

# pick song
song = testing_data[0]

# parameters
input_size = Dataset.num_keys()
hidden_size = 64
num_tags = Dataset.num_tags()
num_layers = 1

# load best model
model = Specialist(input_size, hidden_size, num_tags, num_layers) 
model.load_state_dict(torch.load('saved_models/specialist_BCE_1e-2_999.pth')) 

# generate cherrypicked overfitted Specialist
print "Bach=0, Brahms=1, debussy=2, mozart=3"
print "composer number:", song[1].item()
datapreparation.gen_music_seconds(model,init=song[0],composer=song[1],fs=5,gen_seconds=60,init_seconds=10)


# In[15]:


model = Generalist(input_size, hidden_size, num_layers)
model.load_state_dict(torch.load('saved_models/generalist_BCE_1e-2_208.pth')) # 208,999

# generate cherrypicked overfitted Specialist
print "Bach=0, Brahms=1, debussy=2, mozart=3"
print "composer number:", song[1].item()
datapreparation.gen_music_seconds(model,init=song[0],composer=song[1],fs=5,gen_seconds=60,init_seconds=10)

