
from torch.utils.data import Dataset
import datapreparation as datp
import torch
import numpy as np
import random

class pianoroll_dataset_batch(Dataset):
    """
    
    """
    def __init__(self, root_dir, transform=None, name_as_tag=True,binarize=True, seed=1):
        """
        Args:
            root_dir (string): Directory with all the csv
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(pianoroll_dataset_batch, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.seed = seed
        if(name_as_tag):
            self.tags =  datp.load_all_dataset_names(self.root_dir)
            self.tags_ids=dict(zip(np.unique(self.tags),range(np.unique(self.tags).size)))
        self.data = datp.load_all_dataset(self.root_dir,binarize)
        self.data, self.tags = self.shuffle(self.data, self.tags, seed)
    
    def shuffle(self, list_a, list_b, seed):
        zipped_list = list(zip(list_a, list_b))
        random.Random(seed).shuffle(zipped_list)
        list_a, list_b = zip(*zipped_list)
        return list_a, list_b
    
    def split_datasets(self, split=[0.7,0.85, 1.0]):
        """            datasets.append([map(self.transpose_and_expand_dim,data_shuffled[prev_split_nr:split_nr]),
                           map(self.from_names_to_ids, self.tags_shuffled[prev_split_nr:split_nr]),
                           map(lambda x: self.transpose_and_expand_dim(x,targets=True),data_shuffled[prev_split_nr:split_nr])])"""
        
        split_int = [int(round(len(self.tags)*perc)) for perc in split]
        datasets = [None]*3

        prev_split_nr = 0
        for dataset_nr, split_nr in enumerate(split_int):
            datasets[dataset_nr] = []
            for index in range(prev_split_nr,split_nr):
                datasets[dataset_nr].append(self[index])
                
            prev_split_nr = split_nr
        
        training, validation, testing = datasets[0], datasets[1], datasets[2]
        print("Number of songs in training_data: {}".format(len(training)))
        print("Number of songs in validation_data: {}".format(len(validation)))
        print("Number of songs in testing_data: {}".format(len(testing)))
        return training, validation, testing
    
    def transpose_and_expand_dim(self, data, targets=True):
        if targets:
            return one_end(torch.Tensor(data.T).unsqueeze(1))
        return torch.Tensor(data.T).unsqueeze(1)
    
    def from_names_to_ids(self, name):
        return torch.LongTensor([ self.tags_ids[name]]).unsqueeze(1)

    def gen_batch(self,batchsize=100,chunks_per_song=20):
        return None
    
    # Shouldn't this return the length of the samples of the dataset? not the number of artist...
    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        input_tensor = torch.Tensor(self.data[idx].T).unsqueeze(1)
        tag_tensor = torch.LongTensor([ self.tags_ids[self.tags[idx]] ]).unsqueeze(1)
        output_tensor = one_end(input_tensor)
        return input_tensor, tag_tensor, output_tensor
    
    def set_tags(self,lst_tags):
        self.tags = lst_tags
        
    def num_tags(self):
        return len(self.tags_ids)
    
    def num_keys(self):
        return datp.get_numkeys(self.data)[0]
        
    def view_pianoroll(self,idx):
        datp.visualize_piano_roll(self[idx])
        
class pianoroll_dataset_chunks(Dataset):
    def __init__(self, root_dir,transform=None,binarize=True,delta=1):
        """
        Args:
            root_dir (string): Directory with all the csv
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(pianoroll_dataset_chunks, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.tags =  datp.load_all_dataset_names(self.root_dir)
        self.tags_ids=dict(zip(np.unique(self.tags),range(np.unique(self.tags).size)))
        self.fulldata = datp.load_all_dataset(self.root_dir,binarize)
        self.fulldata = tuple(self.convert_fulldata(i,delta) for i in range(len(self.tags)))
        self.indexes = [(0,0)]
        
    def gen_batch(self,batchsize=100,chunks_per_song=20):
        self.batchsize=batchsize
        self.chunks_per_song=chunks_per_song
        len_full = len(self.tags)
        indexes=zip(np.repeat(np.arange(len_full),chunks_per_song),\
                         np.array([np.arange(chunks_per_song)]*len_full).flatten())
        self.indexes = [indexes[x] for x in np.random.choice(xrange(len(indexes)),batchsize)]
        
    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, idx):
        idx=self.indexes[idx]
        input_tensor = self.fulldata[idx[0]][0].chunk(self.chunks_per_song)[idx[1]]
        output_tensor = self.fulldata[idx[0]][-1].chunk(self.chunks_per_song)[idx[1]]
        tag_tensor = self.fulldata[idx[0]][1]
        return input_tensor, tag_tensor, output_tensor

    def convert_fulldata(self, idx,delta):
        input_tensor = torch.Tensor(self.fulldata[idx].T).unsqueeze(1)
        tag_tensor = torch.LongTensor([ self.tags_ids[self.tags[idx]] ]).unsqueeze(1)
        output_tensor = one_end(input_tensor,delta)
        return input_tensor, tag_tensor, output_tensor
    
    def set_tags(self,lst_tags):
        self.tags = lst_tags
        
    def num_tags(self):
        return len(self.tags_ids)
    
    def num_keys(self):
        return datp.get_numkeys(self.fulldata)[0]
        
    def view_pianoroll(self,idx):
        datp.visualize_piano_roll(self[idx])
        
def one_end(input_tensor,k=1):
    return torch.cat( (input_tensor[k:], torch.zeros(size=(k,input_tensor.shape[1],input_tensor.shape[2]))))