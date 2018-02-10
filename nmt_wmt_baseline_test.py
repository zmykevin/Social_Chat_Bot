"""
This program implements a fundamental sequence to sequence neural machine translation model as the basline for the machine translation vision research. 
"""

import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from preprocessing import *
from encoder import *
from decoder import *
from bleu import *
from evaluate import *

import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

use_cuda = torch.cuda.is_available()
print("Whether GPU is available: {}".format(use_cuda))


#Define the train function
teacher_forcing_ratio = 0.5
clip = 5.0
#The token index for the start of the sentence
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 50 #We will abandon any sentence that is longer than this length

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


## Helper Functions to Print Time Elapsed and Estimated Time Remaining, give the current time and progress
def as_minutes(s):
    m = math.floor(s/60)
    s-= m*60
    return '%dm %ds'%(m,s)
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es-s
    return '%s (- %s)'%(as_minutes(s),as_minutes(rs))

def save_plot(points,save_path,y_label):
    plt.plot(points)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.clf()

def save_plot_compare(point_1, point_2,label_1,label_2,x_axis, save_path, y_label):
    plt.plot(x_axis,point_1,label=label_1)
    plt.plot(x_axis,point_2,label=label_2)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.clf()

#Load the Dataset
data_path = '/home/zmykevin/machine_translation_vision/dataset/NMT_WMT_2017_V2'
trained_model_output_path = '/home/zmykevin/machine_translation_vision/code/mtv_trained_model/WMT17/nmt_wmt_baseline_2'
#trained_model_output_path = '/home/kevin/Kevin/Research/machine_translation_vision/code/trained_model/nmt_wmt_baseline_1'
source_language = 'de'
target_language = 'en'
dataset_suffix = '_WMT_2017'

#Create the directory for the trained_model_output_path
if not os.path.isdir(trained_model_output_path):
    os.mkdir(trained_model_output_path)

#Load the training dataset
train_source = load_data(os.path.join(data_path,'train'+dataset_suffix+'.'+source_language))
train_target = load_data(os.path.join(data_path,'train'+dataset_suffix+'.'+target_language))

print('The size of Training Source and Training Target is: {},{}'.format(len(train_source),len(train_target)))

#Load the validation dataset
val_source = load_data(os.path.join(data_path,'val'+dataset_suffix+'.'+source_language))
val_target = load_data(os.path.join(data_path,'val'+dataset_suffix+'.'+target_language))
print('The size of Validation Source and Validation Target is: {},{}'.format(len(val_source),len(val_target)))

#Load the test dataset
test_source = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+source_language))
test_target = load_data(os.path.join(data_path,'test'+dataset_suffix+'.'+target_language))
print('The size of Test Source and Test Target is: {},{}'.format(len(test_source),len(test_target)))


#Creating List of pairs in the format of [[en_1,de_1], [en_2, de_2], ....[en_3, de_3]]
train_data = [[x.strip(),y.strip()] for x,y in zip(train_source,train_target)]
val_data = [[x.strip(),y.strip()] for x,y in zip(val_source,val_target)]
test_data = [[x.strip(),y.strip()] for x,y in zip(test_source,test_target)]


#Filter the data
train_data = data_filter(train_data,MAX_LENGTH)
val_data = data_filter(val_data,MAX_LENGTH)
test_data = data_filter(test_data,MAX_LENGTH)

print("The size of Training Data after filtering: {}".format(len(train_data)))
print("The size of Val Data after filtering: {}".format(len(val_data)))
print("The size of Test Data after filtering: {}".format(len(test_data)))

#Load the Vocabulary File and Create Word2Id and Id2Word dictionaries for translation
vocab_source = load_data(os.path.join(data_path,'vocab.'+source_language))
vocab_target = load_data(os.path.join(data_path,'vocab.'+target_language))

#Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab_source)
t_word2id, t_id2word = construct_vocab_dic(vocab_target)

print("The vocabulary size for soruce language: {}".format(len(s_word2id)))
print("The vocabulary size for target language: {}".format(len(t_word2id)))

#Generate Train, Val and Test Indexes pairs
train_data_index = create_data_index(train_data,s_word2id,t_word2id)
val_data_index = create_data_index(val_data,s_word2id,t_word2id)
test_data_index = create_data_index(test_data,s_word2id,t_word2id)

val_y_ref = [[d[1].split()] for d in val_data]
test_y_ref = [[d[1].split()] for d in test_data]

'''
print("Done with load and preprocessing data")
print(val_data[10][0])
print(val_data[10][1])
print(val_data_index[10][0])
print(val_data_index[10][1])
print(val_y_ref[10])
'''


#Train the Network. 
attn_model = 'general'
embedding_size = 512
hidden_size = 1024
n_layers = 1
dropout_p = 0.05
bi = True
batch_size = 40
eval_batch_size = 40
batch_num = math.floor(len(train_data_index)/batch_size)
learning_rate = 0.0001
#configuring training parameters
n_epochs = 50
plot_every = 100
print_every = 100
eval_every = 5*print_every
save_every = 2000

#Initialize models
input_size = len(s_word2id)+1
output_size = len(t_word2id)+1
encoder = WMTEncoder(input_size, embedding_size,hidden_size, n_layers)
decoder = WMTDecoder(output_size, embedding_size,2*hidden_size,n_layers,dropout_p)
#encoder = torch.load(os.path.join(trained_model_output_path,"nmt_baseline_trained_encoder.pt"))
#decoder = torch.load(os.path.join(trained_model_output_path,"nmt_baseline_trained_decoder.pt"))
#decoder = AttnDecoder(hidden_size,hidden_size,output_size)
#Move models to GPU
if use_cuda:
    print("Move Models to GPU")
    encoder.cuda()
    decoder.cuda()


#Initialize optimization and criterion    
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(reduce=False)

#Define a learning rate optimizer
encoder_lr_decay_scheduler= optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=0.5,patience=10)
decoder_lr_decay_scheduler= optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer,factor=0.5,patience=10)

#keep track of time elapsed and running averages
start = time.time()
plot_losses = []
plot_val_losses = []
print_loss_total = 0 #Reset every print_every
plot_loss_total = 0 #Reset every plot every
plot_val_bleu = []
plot_test_bleu = []

print(encoder)
print(decoder)
'''
val_translations = []
#Evaluate the Performance
for val_x,val_y,val_x_lengths,val_y_lengths,val_sorted_index in data_generator(train_data_index,eval_batch_size):
    val_translation,val_loss = evaluate_val_2(val_x,val_x_lengths,val_y,val_y_lengths,encoder,decoder,criterion)
    #Reorder val_translations and convert them back to words
    val_translation_reorder = translation_reorder(val_translation,val_sorted_index,t_id2word)
    val_translations += val_translation_reorder
    #val_losses += val_loss
    print(val_loss)
val_output = [' '.join(x) for x in val_translations]
print(val_output[45])

with open('wmt17_train_translations.de','w') as f:
    for x in val_output:
        f.write(x+'\n')
'''

#Begin Training
print("Begin Training")

iter_count = 0

for epoch in range(1,n_epochs + 1):
    for batch_x,batch_y,batch_x_lengths,batch_y_lengths,_ in data_generator(train_data_index,batch_size):
        input_variable = batch_x  #B*W_x
        target_variable = batch_y #B*W_y
        #Run the train function
        loss = train(input_variable, target_variable, batch_x_lengths,batch_y_lengths,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if iter_count == 0: 
            iter_count += 1
            continue
        
        if iter_count%print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, iter_count / n_epochs / batch_num), iter_count, iter_count / n_epochs / batch_num * 100, print_loss_avg)
            print(print_summary)

        if iter_count%eval_every == 0:
            #Print the Bleu Score and loss for Dev Dataset
            val_losses = 0
            eval_iters = 0
            val_translations = []
            for val_x,val_y,val_x_lengths,val_y_lengths,val_sorted_index in data_generator(val_data_index,eval_batch_size):
                val_translation,val_loss = evaluate_val_wmt(val_x,val_x_lengths,val_y,val_y_lengths,encoder,decoder,criterion)
                #Reorder val_translations and convert them back to words
                val_translation_reorder = translation_reorder(val_translation,val_sorted_index,t_id2word) 
                val_losses += val_loss
                eval_iters += 1
                val_translations += val_translation_reorder

            val_loss_mean = val_losses/eval_iters
            val_bleu = compute_bleu(val_y_ref,val_translations)
            print("dev_loss: {}, dev_bleu: {}".format(val_loss_mean,val_bleu[0]))

            test_translations = []
            for test_x,test_y,test_x_lengths,test_y_lengths,test_sorted_index in data_generator(test_data_index,eval_batch_size):
                test_translation= evaluate_test_wmt(test_x,test_x_lengths,encoder,decoder)
                #Reorder val_translations and convert them back to words
                test_translation_reorder = translation_reorder(test_translation,test_sorted_index,t_id2word) 
                test_translations += test_translation_reorder
            
            #Compute the test bleu score
            test_bleu = compute_bleu(test_y_ref,test_translations)
            print("test_bleu: {}".format(test_bleu[0]))

            #Randomly Pick a sentence and translate it to the target language. 
            sample_source, sample_ref, sample_output = random_sample_display(test_data,test_translations)
            print("An example demo:")
            print("src: {}".format(sample_source))
            print("ref: {}".format(sample_ref))
            print("pred: {}".format(sample_output))

        if iter_count%plot_every == 0:
            plot_loss_avg = plot_loss_total / print_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            #Keep the Bleu Score and loss for Dev Dataset
            val_losses = 0
            eval_iters = 0
            val_translations = []
            for val_x,val_y,val_x_lengths,val_y_lengths,val_sorted_index in data_generator(val_data_index,eval_batch_size):
                val_translation,val_loss = evaluate_val_wmt(val_x,val_x_lengths,val_y,val_y_lengths,encoder,decoder,criterion)
                #Reorder val_translations and convert them back to words
                val_translation_reorder = translation_reorder(val_translation,val_sorted_index,t_id2word) 
                val_losses += val_loss
                eval_iters += 1
                val_translations += val_translation_reorder

            val_loss_mean = val_losses/eval_iters
            
            #Schedule a learning rate decay
            encoder_lr_decay_scheduler.step(val_loss_mean)
            decoder_lr_decay_scheduler.step(val_loss_mean)

            #print(val_y_ref[0])
            #print(val_translations[0])
            val_bleu = compute_bleu(val_y_ref,val_translations)
            plot_val_losses.append(val_loss_mean)
            plot_val_bleu.append(val_bleu[0])

            test_translations = []
            for test_x,test_y,test_x_lengths,test_y_lengths,test_sorted_index in data_generator(test_data_index,eval_batch_size):
                test_translation= evaluate_test_wmt(test_x,test_x_lengths,encoder,decoder)
                #Reorder val_translations and convert them back to words
                test_translation_reorder = translation_reorder(test_translation,test_sorted_index,t_id2word) 
                test_translations += test_translation_reorder
            
            #Compute the test bleu score
            test_bleu = compute_bleu(test_y_ref,test_translations)
            plot_test_bleu.append(test_bleu[0])
        
        if iter_count%save_every == 0:
            #Save the model every save_every iterations.
            torch.save(encoder,os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_coder_{}.pt'.format(iter_count)))
            torch.save(decoder,os.path.join(trained_model_output_path,'nmt_wmt_baseline_trained_coder_{}.pt'.format(iter_count)))
        iter_count += 1

print("Training is done.")
#Save the Model
print("Saving Model ...")


#Save the Model
torch.save(encoder,os.path.join(trained_model_output_path,"nmt_wmt_baseline_trained_encoder.pt"))
torch.save(decoder,os.path.join(trained_model_output_path,"nmt_wmt_baseline_trained_decoder.pt"))

#Save the hyperparameters
hyprams={"teacher_forcing_ratio":teacher_forcing_ratio,
"clip":clip,
"SOS":SOS_token,
"EOS":EOS_token,
"UNK":UNK_token,
"MAX_LENGTH":MAX_LENGTH,
"data_dir":data_path,
"output_path":trained_model_output_path,
"src":source_language,
"tgt":target_language,
"dataset_suffix":dataset_suffix,
"attn_model":attn_model,
"hidden_size":hidden_size,
"n_layers":n_layers,
"dropout":dropout_p,
"bi":bi,
"batch_size":batch_size,
"eval_batch_size":eval_batch_size,
"learning_rate":learning_rate,
"epochs":n_epochs,
}
#Save the hyprams:
torch.save(hyprams,os.path.join(trained_model_output_path,"hyprams"))


#Plot and Save the Loss, and Bleu Score Plot
plot_loss_save_path = os.path.join(trained_model_output_path,"nmt_wmt_baseline_train_val_loss_plot.png")
plot_bleu_save_path = os.path.join(trained_model_output_path,"nmt_wmt_baseline_val_test_bleu_plot.png")

#Create the x_axis list which represents ths number of iters
iter_progress = [plot_every*(i+1) for i in range(0,len(plot_losses))]
save_plot_compare(plot_losses,plot_val_losses,'train_losses','val_losses',iter_progress,plot_loss_save_path,'loss')
save_plot_compare(plot_val_bleu,plot_test_bleu,'val_bleu','test_bleu',iter_progress,plot_bleu_save_path,'bleu')
