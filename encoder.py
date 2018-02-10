import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math

use_cuda = torch.cuda.is_available()

#Define the Encoder Net Structure
class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size, n_layers=1, bi=False):
        #Initialize the Super Class
        super(Encoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if bi:
            self.bi = True
            self.n_direction = 2
        else:
            self.n_direction = 1
            self.bi = False

        #Define the Embedding Layer
        self.embedding = nn.Embedding(input_size,hidden_size) #Later We want to initialize it with Word2Vec
        #Define the LSTM Cells
        self.lstm = nn.LSTM(hidden_size,hidden_size,num_layers=n_layers,bidirectional=bi)
    def forward(self,input_var,input_lengths,hidden):
        """
        Input Variable:
            input_var: A variables whose size is (B,W), B is the batch size and W is the longest sequence length in the batch 
            input_lengths: The lengths of each element in the batch. 
            hidden: The hidden state variable whose size is (num_layer*num_directions,batch_size,hidden_size)
        Output:
            output: A variable with tensor size W*B*N, W is the maximum length of the batch, B is the batch size, and N is the hidden size
            hidden: The hidden state variable with tensor size (num_layer*num_direction,B,N)
        """
        #Convert input sequence into a pack_padded tensor
        embedded_x = self.embedding(input_var).transpose(0,1) #The dimension of embedded_x is  W*B*N, where N is the embedding size.
        
        #Get a pack_padded sequence
        embedded_x = torch.nn.utils.rnn.pack_padded_sequence(embedded_x,input_lengths)
        
        #Get an output pack_padded sequence
        output,hidden = self.lstm(embedded_x,hidden) 
        #Unpack the pack_padded sequence
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output) #The size of output will be W*B*N
        
        if self.bi:
            #Sum bidirectional outputs
            output = output[:,:,:self.hidden_size]+output[:,:,self.hidden_size:]
        
        return output,hidden
    
    def init_hidden(self,batch_size=1):
        result = (Variable(torch.zeros(self.n_layers*self.n_direction,batch_size,self.hidden_size)),Variable(torch.zeros(self.n_layers*self.n_direction,batch_size,self.hidden_size)))
        if use_cuda:
            result = (result[0].cuda(),result[1].cuda())
        return result


#Define the Encoder for WMTBaseline_Model
class WMTEncoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size, n_layers=1):
        #Initialize the Super Class
        super(WMTEncoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_direction = 2
        #Define the Embedding Layer
        self.embedding = nn.Embedding(input_size,embedding_size) #Later We want to initialize it with Word2Vec
        #Define the LSTM Cells
        self.gru = nn.GRU(embedding_size,hidden_size,num_layers=n_layers,bidirectional=True)
    def forward(self,input_var,input_lengths,hidden):
        """
        Input Variable:
            input_var: A variables whose size is (B,W), B is the batch size and W is the longest sequence length in the batch 
            input_lengths: The lengths of each element in the batch. 
            hidden: The hidden state variable whose size is (num_layer*num_directions,batch_size,hidden_size)
        Output:
            output: A variable with tensor size W*B*N, W is the maximum length of the batch, B is the batch size, and N is the hidden size
            hidden: The hidden state variable with tensor size (num_layer*num_direction,B,N)
        """
        #Convert input sequence into a pack_padded tensor
        embedded_x = self.embedding(input_var).transpose(0,1) #The dimension of embedded_x is  W*B*N, where N is the embedding size.
        
        #Get a pack_padded sequence
        embedded_x = torch.nn.utils.rnn.pack_padded_sequence(embedded_x,input_lengths)
        
        #Get an output pack_padded sequence
        output,hidden = self.gru(embedded_x,hidden) 
        #Unpack the pack_padded sequence
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output) #The size of output will be W*B*N
        
        return output,hidden
    
    def init_hidden(self,batch_size=1):
        result = Variable(torch.zeros(self.n_layers*self.n_direction,batch_size,self.hidden_size))
        if use_cuda:
            result = result.cuda()
        return result