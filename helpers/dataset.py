from torch.utils.data import Dataset
import datapreparation as datp
import torch
import numpy as np

class pianoroll_dataset_batch(Dataset):
    """
    
    """
    def __init__(self, root_dir, transform=None, name_as_tag=True,binarize=True):
        """
        Args:
            root_dir (string): Directory with all the csv
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(pianoroll_dataset_batch, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        if(name_as_tag):
            self.tags =  datp.load_all_dataset_names(self.root_dir)
            self.tags_ids=dict(zip(np.unique(self.tags),range(np.unique(self.tags).size)))
        self.data = datp.load_all_dataset(self.root_dir,binarize)

    def gen_batch(self,batchsize=100,chunks_per_song=20):
        return None
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
    return torch.cat( (input_tensor[k:], torch.zeros(size=(k,input_tensor.shape[1],input_tensor.shape[2]))) )