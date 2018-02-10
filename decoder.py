import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math

use_cuda = torch.cuda.is_available()
print("Whether GPU is available: {}".format(use_cuda))
MAX_LENGTH = 40

class Attn(nn.Module):
    def __init__(self,method,hidden_size,max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat': #Concat is still not correctly implemented for batch_size > 1. 1/14/2018: 6:13 AM
            self.attn = nn.Linear(self.hidden_size*2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1,hidden_size))
    def forward(self,hidden,encoder_outputs):
        #print("the hidden matrix size is: {}".format(hidden.size()))
        seq_len,batch_size = encoder_outputs.size()[0],encoder_outputs.size()[1] #Get the sequence length
        #Create variable to store attention energies
        attn_energies = Variable(torch.zeros(batch_size,seq_len)) # Batch_Size*SeqLength
        if use_cuda:
            attn_energies = attn_energies.cuda()
        #Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[:,i] = self.score(hidden[0],encoder_outputs[i])
            
        #Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return F.softmax(attn_energies).unsqueeze(1)
    def score(self,hidden,encoder_output):
        """
        Input:
            hidden: last output of the decoder at step t, with size B*N
            encoder_output: the corresponding encoder output at time step t, with size B*N
        Output:
            energy: The corresponding output energy with size B*1*1
        """
        if self.method == 'dot':
            hidden = hidden.unsqueeze(1)
            encoder_output = encoder_output.unsqueeze(2)
            energy = hidden.bmm(encoder_output)
            #energy = torch.bmm(hidden.unsqueeze(1),encoder_output.unsqueeze(2))
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output).unsqueeze(2)
            hidden = hidden.unsqueeze(1)
            energy = hidden.bmm(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden,encoder_output),1))
            energy = self.other.dot(energy)
            return energy

class Attn_2(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn_2, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        # end of update
        self.softmax = nn.Softmax()

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return self.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttn(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttn, self).__init__()
        self.hidden_size = hidden_size
        #attn_h,attn_e and v are the three parameters to tune. 
        self.attn_h = nn.Linear(self.hidden_size, hidden_size)
        self.attn_e = nn.Linear(self.hidden_size,hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        #Normalize Data
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        # end of update
        self.softmax = nn.Softmax()

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return self.softmax(attn_energies).unsqueeze(1) # normalize with softmax, attn_energies = B * 1 * T

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn_h(hidden)+self.attn_e(encoder_outputs)) #The size of energy is B*T*H
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

#Construct the attention decoder
class AttnDecoder(nn.Module):
    def __init__(self,attn_model,output_size,hidden_size,n_layers=1,dropout_p=0.1):
        super(AttnDecoder,self).__init__()
        
        #Keep Parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        #Define layers
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        self.concat = nn.Linear(hidden_size*2,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        #Pick the attention model
        '''
        if attn_model != 'none':
            self.attn = Attn(attn_model,hidden_size)
        '''
        self.attn = Attn_2(attn_model,hidden_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        Input:
            word_input: The next word that is input into the current decoder step, size: B*1
            last_context: The context vector inherited from last step, size: B*N
            last_hidden: The hidden state inherited from last step, size: num_layer*num_direction*B*N
            encoder_outputs: The output vectors for all steps from encoder, size: W*B*N, W is the length of the encoder state. 
        Output:
            output: The prediction for the decoded output, which has the size (N*B*output_size)
            context: The current context vector from current step, which has the size (B*N)
            hidden: The current hidden state, which has the size (num_layer*num_direction,B,N)
            attn_weights: The current attention weights, which has the size(B,1,W), where W is the longest width of the input sequence of current batch. 
        """
        # Get the embedding of the current input word (last output word)
        batch_size = word_input.size()[0]
        word_embedding = self.embedding(word_input)
        word_embedding = self.embedding_dropout(word_embedding).view(1,batch_size,self.hidden_size)#S = 1*B*N
        
        #print(word_embedding.size())
        # Combine embedded input word and last context, run through RNN
        #rnn_input = torch.cat((word_embedding, last_context.unsqueeze(0)),2)
        rnn_output, hidden = self.lstm(word_embedding, last_hidden)
        
        #Calculate attention from current RNN state and all encoder outputs; apply to encoder output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1)) # context size: B*1*N
        
        #Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) #S = 1*B*N -> B*N
        context = context.squeeze(1) #S = B*1*N -> B*N
        concat_input = torch.cat((rnn_output,context),1)
        concat_output = F.tanh(self.concat(concat_input))
        
        #Finally predict next token
        output = F.log_softmax(self.out(concat_output))
        #output = self.out(concat_output)
        #Return Final Output, Hidden State, and Attention Weights (for visualization)
        return output, hidden, attn_weights

#Implement of cGRU from nematus language toolkit paper address is: 
class WMTDecoder(nn.Module):
    def __init__(self,output_size,embedding_size,hidden_size,n_layers=1,dropout_p=0.1):
        super(WMTDecoder,self).__init__()

        #Keep Parameters for Reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size  
        self.n_layers=n_layers
        self.dropout_p=0.1


        #Define layers
        self.embedding=nn.Embedding(output_size,embedding_size)
        self.embedding_dropout = nn.Dropout(dropout_p)
        #The Three main components for the cGRU.
        self.gru_1 = nn.GRU(embedding_size,hidden_size,bias=False,num_layers=n_layers,dropout=dropout_p)
        self.attn = BahdanauAttn(hidden_size)
        self.gru_2 = nn.GRU(hidden_size,hidden_size,bias=False,num_layers=n_layers,dropout=dropout_p)
        #Three matrix to generate a intermediate representation tj for final output
        self.W1 = nn.Linear(hidden_size,hidden_size)
        self.W2 = nn.Linear(embedding_size,hidden_size)
        self.W3 = nn.Linear(hidden_size,hidden_size)
        #Output Layer
        self.out = nn.Linear(hidden_size,output_size)
    def forward(self,word_input,last_hidden,encoder_outputs):
        '''
        Input:
            word_input: A tensor with size B*1, representing the previous predicted word 
            last_hidden: The hidden state vector from the previous timestep, s_t_1
            encoder_outputs: Size T_in*B*H_e
        '''
        batch_size = word_input.size()[0]
        #Embedding Word input to WordVectors
        word_embedding = self.embedding(word_input)
        word_embedding = self.embedding_dropout(word_embedding).view(1,batch_size,-1) #The size for wordembedding is 1*B*E, E is the Embedding_Size

        #Process the word_embedding through the first gru to generate the intermediate representation
        gru_1_output,gru_1_hidden = self.gru_1(word_embedding,last_hidden) #The gru_1_hidden is the intermediate hidden state, with size(L,B,N), N is the hidden_size, L is the layer_size
        
        #Compute the Attentional Weights Matrix
        attn_weights = self.attn(gru_1_output[-1],encoder_outputs)
        #Get the update context
        context = attn_weights.bmm(encoder_outputs.transpose(0,1)) # context size B*1*H
        #Compute the output from second gru. 
        gru_2_output,gru_2_hidden = self.gru_2(context.transpose(0,1),gru_1_hidden)

        #Squeeze the Size
        gru_2_output = gru_2_output.squeeze(0) #1*B*H -> B*H
        context = context.squeeze(1) #B*1*H -> B*H
        word_embedding = word_embedding.squeeze(0) #1*B*E -> B*E
        #Compute the intermediate representation before softmax
        concat_output = F.tanh(self.W1(gru_2_output)+self.W2(word_embedding)+self.W3(context))

        output = F.log_softmax(self.out(concat_output))

        return output,gru_2_hidden