# Composers of music based on RNN structure
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generalist(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Generalist, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.notes_encoder = nn.Linear(in_features=input_size,
                                       out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.notes_decoder = nn.Linear(in_features=hidden_size,
                                      out_features=input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_sequence, tag, hidden=None):
        
        enc_notes = self.notes_encoder(input_sequence)
        """
        # Run rnns on non-padded regions of the batch
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded, input_seq_lengths)
        outputs, hidden = self.GRU(packed_seq, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # back to padded format
        """

        outputs, hidden = self.gru(enc_notes, hidden)
        
        outputs = self.sigmoid(self.notes_decoder(outputs))
        return outputs, hidden


class Specialist(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, num_layers):
        super(Specialist, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.num_layers = num_layers
        
        self.tags_embedding = nn.Embedding(num_tags, hidden_size)
        self.notes_encoder = nn.Linear(in_features=input_size,
                                       out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.notes_decoder = nn.Linear(in_features=hidden_size,
                                      out_features=input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_sequence, tag, hidden=None):
        if (hidden is None):
            hidden = self.tags_embedding(tag)
            hidden = torch.cat(tuple((hidden for i in range(self.num_layers))))
            enc_input = self.notes_encoder(input_sequence)
        
        
        output, hidden = self.gru(self.notes_encoder(input_sequence), hidden)
        output = self.sigmoid(self.notes_decoder(output))
        return output, hidden
