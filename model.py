import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import math
import random
from encoder import WMTEncoder
from decoder import WMTDecoder
from preprocessing import *

use_cuda = torch.cuda.is_available()
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 50


class ImagineSeq2Seq(nn.Module):
    def __init__(self, \
                 src_size, \
                 tgt_size, \
                 im_feats_size, \
                 src_embedding_size, \
                 tgt_embedding_size, \
                 hidden_size, \
                 shared_embedding_size, \
                 n_layers=1, \
                 dropout_p=0.05):
        super(ImagineSeq2Seq,self).__init__()
        #Define all the parameters
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.im_feats_size = im_feats_size
        self.src_embedding_size = src_embedding_size
        self.tgt_embedding_size = tgt_embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = 0.05
        self.shared_embedding_size = shared_embedding_size

        #Define all the parts. 
        self.encoder = WMTEncoder(src_size,src_embedding_size,hidden_size,n_layers)
        self.decoder = WMTDecoder(tgt_size,tgt_embedding_size,2*hidden_size,n_layers,dropout_p)
        #Vision Embedding Layer
        self.im_embedding = nn.Linear(im_feats_size,shared_embedding_size)
        self.text_embedding = nn.Linear(2*hidden_size,shared_embedding_size)
        
    def forward(self,src_var,src_lengths,im_var,teacher_force_ratio=0,tgt_var=None,max_length=MAX_LENGTH):
        '''
        Input: 
            src_var: The minibatch input sentence indexes representation with size (B*W_s)
            src_lengths: The list of lenths of each sentence in the minimatch, the size is (B)
            im_var: The minibatch of the paired image ResNet Feature vecotrs, with the size(B*I), I is the image feature size.
            teacher_force_ratio: A scalar between 0 and 1 which defines the probability ot conduct the teacher_force traning.
            tgt_var: The output sentence groundtruth, if provided it will be used to help guide the training of the network. The Size is (B*W_t)
                     If not, it will just generate a target sentence which is shorter thatn max_length or stop when it finds a EOS_Tag.
            max_length: A integer value that specifies the longest sentence that can be generated from this network.     
        Output:            
        '''
        #Define the batch_size and input_length
        src_l = src_var.size()[1]
        batch_size = src_var.size()[0]
        
        #Encoder src_var
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs,encoder_hidden = self.encoder(src_var,src_lengths,encoder_hidden)
        
        #Prepare the Input and Output Variables for Decoder
        decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
        
        if use_cuda:
            decoder_input = decoder_input.cuda()
            
        if tgt_var is not None:
            tgt_l = tgt_var.size()[1]
            outputs = Variable(torch.zeros(tgt_l,batch_size,self.tgt_size))
            if use_cuda:
                outputs = outputs.cuda()
            #Determine whether teacher forcing is used. 
            is_teacher = random.random() < teacher_force_ratio
            if is_teacher: 
                for di in range(tgt_l):
                    decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    #update the outputs
                    outputs[di] = decoder_output
                    decoder_input = tgt_var[:,di]
            else:
                for di in range(tgt_l):
                    decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    #update the outputs
                    outputs[di] = decoder_output
                    _,top1 = decoder_output.data.topk(1)
                    decoder_input = Variable(top1)
                    if use_cuda:
                    	decoder_input = decoder_input.cuda()
        else:
            tgt_l = max_length
            outputs = Variable(torch.zeros(tgt_l,batch_size,self.tgt_size))
            for di in range(tgt_l):
                decoder_output,decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                #update the outputs
                outputs[di] = decoder_output
                _,top1 = decoder_output.data.topk(1)
                decoder_input = Variable(top1)
                if use_cuda:
                	decoder_input = decoder_input.cuda()

        #Embed the Image vector to the shared space
        im_embedding = F.normalize(self.im_embedding(im_var)) #im_embedding: B*(2*hidden_size)
        #im_embedding = F.normalize(im_var)

        #Embed the decoder hidden state to the shared space
        decoder_hidden = decoder_hidden.squeeze(0)
        text_embedding = F.normalize(self.text_embedding(decoder_hidden))
        
        #Compute the Similarity Score.
        s_im_text = im_embedding.matmul(text_embedding.transpose(0,1))
        s_text_im = text_embedding.matmul(im_embedding.transpose(0,1))
        
        return outputs,s_im_text,s_text_im

#Create the Cycle Encoder Decoder Structure. 
class CycleSeq2Seq(nn.Module):
    def __init__(self, \
                 src_size, \
                 tgt_size, \
                 src_embedding_size, \
                 tgt_embedding_size, \
                 hidden_size, \
                 n_layers=1, \
                 dropout_p=0.05, \
                 forward_encoder = None, \
                 forward_decoder = None, \
                 backward_encoder = None, \
                 backward_decoder = None):
        super(CycleSeq2Seq,self).__init__()
        #Define all the parameters
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.src_embedding_size = src_embedding_size
        self.tgt_embedding_size = tgt_embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = 0.05
        
        #Define the forward_encoder, forward_decoder, backward_encder, backward_decoder. 
        if forward_encoder is None:
            self.forward_encoder = WMTEncoder(src_size,src_embedding_size,hidden_size,n_layers)
        else:
            self.forward_encoder = forward_encoder
        if forward_decoder is None:
            self.forward_decoder = WMTDecoder(tgt_size,tgt_embedding_size,2*hidden_size,n_layers,dropout_p)
        else:
            self.forward_decoder = forward_decoder
        if backward_encoder is None:
            self.backward_encoder = WMTEncoder(tgt_size,src_embedding_size,hidden_size,n_layers)
        else:
            self.backward_encoder = backward_encoder
        if backward_decoder is None:
            self.backward_decoder = WMTDecoder(src_size,tgt_embedding_size,2*hidden_size,n_layers,dropout_p)
        else:
            self.backward_decoder = backward_decoder
        
    def forward(self,src_var,src_lengths,tgt_var,teacher_force_ratio=0,max_length=MAX_LENGTH):
        '''
        Input: 
            src_var: The minibatch input sentence indexes representation with size (B*W_s)
            src_lengths: The list of lenths of each sentence in the minimatch, the size is (B)
            im_var: The minibatch of the paired image ResNet Feature vecotrs, with the size(B*I), I is the image feature size.
            teacher_force_ratio: A scalar between 0 and 1 which defines the probability ot conduct the teacher_force traning.
            tgt_var: The output sentence groundtruth, if provided it will be used to help guide the training of the network. The Size is (B*W_t)
                     If not, it will just generate a target sentence which is shorter thatn max_length or stop when it finds a EOS_Tag.
            max_length: A integer value that specifies the longest sentence that can be generated from this network.     
        Output:            
        '''
        #Define the batch_size and input_length
        batch_size = src_var.size()[0]
        src_l = src_var.size()[1]
        tgt_l = tgt_var.size()[1]
        
        #Encoder src_var
        forward_encoder_hidden = self.forward_encoder.init_hidden(batch_size)
        forward_encoder_outputs,forward_encoder_hidden = self.forward_encoder(src_var,src_lengths,forward_encoder_hidden)
        
        #Prepare the Input and Output Variables for Decoder
        forward_decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        forward_decoder_hidden = torch.mean(forward_encoder_outputs,dim=0,keepdim=True)
        
        if use_cuda:
            forward_decoder_input = forward_decoder_input.cuda()
        
        forward_decoded_words = []
        forward_outputs = Variable(torch.zeros(tgt_l,batch_size,self.tgt_size))
        if use_cuda:
            forward_outputs = forward_outputs.cuda()
        #Determine whether teacher forcing is used. 
        is_teacher = random.random() < teacher_force_ratio
        if is_teacher: 
            for di in range(tgt_l):
                forward_decoder_output,forward_decoder_hidden = self.forward_decoder(forward_decoder_input, forward_decoder_hidden, forward_encoder_outputs)
                #update the outputs
                forward_outputs[di] = forward_decoder_output
                #Add the decoded word to the forward_decodede_words
                _,topi = forward_decoder_output.data.topk(1)
                forward_decoded_words.append(topi[:,0])
                forward_decoder_input = tgt_var[:,di]
        else:
            for di in range(tgt_l):
                forward_decoder_output,forward_decoder_hidden = self.forward_decoder(forward_decoder_input, forward_decoder_hidden, forward_encoder_outputs)
                #update the outputs
                forward_outputs[di] = forward_decoder_output
                _,topi = forward_decoder_output.data.topk(1)
                forward_decoded_words.append(topi[:,0])
                forward_decoder_input = Variable(topi)
        
        #Start the backward_encoder_input_variable
        backward_encoder_input_list = []
        for b in range(batch_size):
            current_list = []
            for i in range(tgt_l):
                current_translation_token = forward_decoded_words[i][b]
                current_list.append(current_translation_token)
                if current_translation_token == EOS_token:
                    break
            backward_encoder_input_list.append(current_list)
        #Convert backward_encoder_input_list to a varibale, and also the backward_reoder_list
        backward_encoder_input_variable,backward_encoder_input_lengths,backward_reorder_list = data_generator_single(backward_encoder_input_list) 
        #Reorde the input_variable to serve as the guidance for the second network
        input_variable_reorder = torch.zeros_like(src_var)
        for i in range(batch_size):
            input_variable_reorder[i] = src_var[backward_reorder_list[i]]

        #Run Words through Encoder
        #Inheritate the hidden states from Decoder
        backward_encoder_hidden = forward_decoder_hidden.view(2,batch_size,-1) #Can be a issue when we have a bidirectional encoder
        backward_encoder_outputs, backward_encoder_hidden = self.backward_encoder(backward_encoder_input_variable, backward_encoder_input_lengths, backward_encoder_hidden)
        
        #Prepare input and output variables
        backward_decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
        #backward_decoder_hidden = backward_encoder_hidden
        backward_decoder_hidden = torch.mean(backward_encoder_outputs,dim=0,keepdim=True)
        
        #Initialize the backward output
        backward_outputs = Variable(torch.zeros(src_l,batch_size,self.src_size))
        
        if use_cuda:
            backward_decoder_input = backward_decoder_input.cuda()
            #input_variable_reorder=input_variable_reorder.cuda()
            backward_outputs = backward_outputs.cuda()
        
        #Start to decode
        if is_teacher:
            for di in range(src_l):
                backward_decoder_output, backward_decoder_hidden = self.backward_decoder(backward_decoder_input, backward_decoder_hidden, backward_encoder_outputs)
                backward_decoder_input = input_variable_reorder[:,di] #Current Target is the next input
                backward_outputs[di] = backward_decoder_output
        else:
            for di in range(src_l):
                backward_decoder_output, backward_decoder_hidden = self.backward_decoder(backward_decoder_input, backward_decoder_hidden, backward_encoder_outputs)
                backward_outputs[di] = backward_decoder_output
                #print(forward_decoder_output.size())
                #Get the predicted output word
                _,topi = backward_decoder_output.data.topk(1)
                backward_decoder_input = Variable(topi) # Next target is next input
                if use_cuda:
                    backward_decoder_input = backward_decoder_input.cuda()
        
        backward_outputs_reorder = torch.zeros_like(backward_outputs)
        for i in range(batch_size):
            backward_outputs_reorder[:,backward_reorder_list[i],:] = backward_outputs[:,i,:]
            
        #Return the farward and backward outputs
        return forward_outputs,backward_outputs_reorder

