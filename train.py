import torch
from torch.autograd import Variable
import torch.nn as nn
import random
from random import randint
import torch.nn.functional as F
import math

use_cuda = torch.cuda.is_available()

clip = 5.0
teacher_forcing_ratio = 0.5
#The token index for the start of the sentence
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 40 #We will abandon any sentence that is longer than this length

def train(input_variable, target_variable, input_lengths,target_lengths,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #Make model back to trianing
    encoder.train()
    decoder.train()
    #Zero gradients of both optimizerse
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word
    
    #Get size of input and target sentences
    input_length = input_variable.size()[1]
    target_length = target_variable.size()[1]
    batch_size = input_variable.size()[0]

    #Run Words through Encoder
    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths,encoder_hidden)#Encoder outputs has the size T*B*2*N
    #Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
    #Initialize the hidden state of decoder as the mean of the encoder annotaiton. 
    decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
    
    target_lengths_var = Variable(torch.FloatTensor(target_lengths))
    if use_cuda:
        decoder_input = decoder_input.cuda()
        target_lengths_var = target_lengths_var.cuda()
        #all_decoder_outputs.cuda()

    #Run teacher forcing only during training.
    is_teacher = random.random() < teacher_forcing_ratio
    if is_teacher: 
        for di in range(target_length):
            decoder_output,decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss_n = criterion(decoder_output,target_variable[:,di])
            loss += torch.mean(torch.div(loss_n,target_lengths_var))
            decoder_input = target_variable[:,di]
    else:
        for di in range(target_length):
            decoder_output,decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss_n = criterion(decoder_output,target_variable[:,di])
            loss += torch.mean(torch.div(loss_n,target_lengths_var))
            _,top1 = decoder_output.data.topk(1)
            decoder_input = Variable(top1)
            if use_cuda:
                decoder_input = decoder_input.cuda()
        '''
        decoder_input = target_variable[:,di] # Next target is next input
        '''
    #Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    
    #Optimize the Encoder and Decoder
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0]

def evaluate_val_wmt(input_variable,input_lengths,target_variable,target_lengths,encoder, decoder, criterion,max_length=MAX_LENGTH):
    """
    This function provides a predicted transfomation of the input_variable. 
    Process the input_varible as a whole.
    Input:
        input_variable: size(B,W_x)
    Output:
        output: list of list of words, where each list of words is the predicted translation from the model
    """
    encoder.eval()
    decoder.eval()
    input_length = input_variable.size()[1]
    target_length = target_variable.size()[1]
    batch_size = input_variable.size()[0]
    loss = 0

    #Run trough encoder
    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, encoder_hidden)

    #Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
    decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
    
    #all_decoder_outputs = Variable(torch.zeros(target_length, batch_size, decoder.output_size))
    target_lengths_var = Variable(torch.FloatTensor(target_lengths))
    if use_cuda:
        decoder_input = decoder_input.cuda()
        #all_decoder_outputs.cuda()
        target_lengths_var = target_lengths_var.cuda()

    decoder_translation_list = []
    for di in range(target_length):
        decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss_n = criterion(decoder_output,target_variable[:,di])
        loss+= torch.mean(torch.div(loss_n,target_lengths_var))
        #Get most likely word index (highest value from input
        topv, topi = decoder_output.data.topk(1)
        #ni = topi[0][0]
        ni = topi[:,0] #Size B*1

        decoder_input = Variable(topi) #The chosen word is the next input
        if use_cuda:
            decoder_input = decoder_input.cuda()
        # Stop at end of sentence (not necessary when usin known targets)
        #if ni == EOS_token: break
        decoder_translation_list.append(ni)

    #Transform the translation result to list of list
    output = []
    for b in range(batch_size):
        current_list = []
        for i in range(target_length):
            current_translation_token = decoder_translation_list[i][b]
            if current_translation_token == EOS_token:
                break
            current_list.append(current_translation_token)
        output.append(current_list)
    return output,loss.data[0]

def evaluate_test_wmt(input_variable,input_lengths,encoder, decoder, max_length=MAX_LENGTH):
    """
    This function provides a predicted transfomation of the input_variable. 
    Process the input_varible as a whole.
    Input:
        input_variable: size(B,W_x)
    Output:
        output: list of list of words, where each list of words is the predicted translation from the model
    """
    encoder.eval()
    decoder.eval()
    input_length = input_variable.size()[1]
    batch_size = input_variable.size()[0]

    #Run trough encoder
    encoder_hidden = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, encoder_hidden)

    #Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[SOS_token] for x in range(batch_size)]))
    decoder_hidden = torch.mean(encoder_outputs,dim=0,keepdim=True)
    
    if use_cuda:
        decoder_input = decoder_input.cuda()
    
    decoder_translation_list = []
    for di in range(max_length):
            decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            #Get most likely word index (highest value from input
            topv, topi = decoder_output.data.topk(1)
            ni = topi[:,0] #Size B*1

            decoder_input = Variable(topi) #The chosen word is the next input
            if use_cuda:
                decoder_input = decoder_input.cuda()
            decoder_translation_list.append(ni)

    #Transform the translation result to list of list
    output = []
    for b in range(batch_size):
        current_list = []
        for i in range(max_length):
            current_translation_token = decoder_translation_list[i][b]
            #current_list.append(current_translation_token)
            if current_translation_token == EOS_token:
                break
            current_list.append(current_translation_token)
        output.append(current_list)
    return output

def random_sample_display(test_data,output_list):
    sample_index = randint(0,len(test_data)-1)
    sample_source = test_data[sample_index][0]
    sample_ref = test_data[sample_index][1]
    sample_output_tokens = output_list[sample_index]
    sample_output = ' '.join(sample_output_tokens)
    return sample_source, sample_ref, sample_output
