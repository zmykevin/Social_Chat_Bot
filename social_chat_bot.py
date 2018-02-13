'''
A pytorch chatbot
'''

import os
from preprocessing import *
from encoder import *
from decoder import *
from train import *
from bleu import *

#Load the two dataset
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 50 #We will abandon any sentence that is longer than this length


#Load the Dataset
data_path = './'
trianed_model_output_path = '/home/zmykevin/alexa_challenge/social_bot/trained_model/chatbot_1'

X = load_data(os.path.join(data_path,'first.txt'))
Y = load_data(os.path.join(data_path,'second.txt'))

#Split X and Y into train and val
train_source,val_source,test_source = X[:720000],X[720000:721000],X[721000:]
train_target,val_target,test_target = X[:720000],Y[720000:721000],Y[721000:]

#Create the train, val and test dataset
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
#vocab = load_data(os.path.join(data_path,'vocab.en'))
#Create a vocab from files
vocab = []
for x in X:
	for w in x.split():
		if w not in vocab:
			vocab.append(w)
print(len(vocab))

'''
#Construct the source_word2id, source_id2word, target_word2id, target_id2word dictionaries
s_word2id, s_id2word = construct_vocab_dic(vocab)

print("The vocabulary size for soruce language: {}".format(len(s_word2id)))

#Generate Train, Val and Test Indexes pairs
train_data_index = create_data_index(train_data,s_word2id,s_word2id)
val_data_index = create_data_index(val_data,s_word2id,s_word2id)
test_data_index = create_data_index(test_data,s_word2id,s_word2id)

print(train_data_index[0])
print(train_data_index[1])

val_y_ref = [[d[1].split()] for d in val_data]
test_y_ref = [[d[1].split()] for d in test_data]

#Construct the Model
#Train the Network. 
attn_model = 'general'
embedding_size = 300
hidden_size = 256
n_layers = 1
dropout_p = 0.05
batch_size = 128
eval_batch_size = 128
batch_num = math.floor(len(train_data_index)/batch_size)
learning_rate = 0.0001
#configuring training parameters
n_epochs = 1
print_every = 10
eval_every = 5*print_every
save_every = 2000

#Initialize models
input_size = len(s_word2id)+1
output_size = len(s_word2id)+1
#Construct the Encoder Decoder
encoder = WMTEncoder(input_size, embedding_size,hidden_size, n_layers)
decoder = WMTDecoder(output_size, embedding_size,2*hidden_size,n_layers,dropout_p)

#Construct the optimizer
if use_cuda:
    print("Move Models to GPU")
    encoder.cuda()
    decoder.cuda()


#Initialize optimization and criterion    
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(reduce=False)

#Define a learning rate optimizer
encoder_lr_decay_scheduler= optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=0.5,patience=5)
decoder_lr_decay_scheduler= optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer,factor=0.5,patience=5)

#Start Training
print_loss_total = 0 #Reset every print_every
iter_count = 0
for epoch in range(1,n_epochs + 1):
    for batch_x,batch_y,batch_x_lengths,batch_y_lengths,_ in data_generator(train_data_index,batch_size):
        input_variable = batch_x  #B*W_x
        target_variable = batch_y #B*W_y
        #Run the train function
        loss = train(input_variable, target_variable, batch_x_lengths,batch_y_lengths,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        
        if iter_count == 0: 
            iter_count += 1
            continue
        
        if iter_count%print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '(%d %d%%) %.4f' % (iter_count, iter_count / n_epochs / batch_num * 100, print_loss_avg)
            print(print_summary)

        if iter_count%eval_every == 0:
            #Print the Bleu Score and loss for Dev Dataset
            val_losses = 0
            eval_iters = 0
            val_translations = []
            for val_x,val_y,val_x_lengths,val_y_lengths,val_sorted_index in data_generator(val_data_index,eval_batch_size):
                val_translation,val_loss = evaluate_val_wmt(val_x,val_x_lengths,val_y,val_y_lengths,encoder,decoder,criterion)
                #Reorder val_translations and convert them back to words
                val_translation_reorder = translation_reorder(val_translation,val_sorted_index,s_id2word) 
                val_losses += val_loss
                eval_iters += 1
                val_translations += val_translation_reorder

            val_loss_mean = val_losses/eval_iters
            val_bleu = compute_bleu(val_y_ref,val_translations)
            print("dev_loss: {}, dev_bleu: {}".format(val_loss_mean,val_bleu[0]))

            #Randomly Pick a sentence and translate it to the target language. 
            sample_source, sample_ref, sample_output = random_sample_display(val_data,val_translations)
            print("An example demo:")
            print("src: {}".format(sample_source))
            print("ref: {}".format(sample_ref))
            print("pred: {}".format(sample_output))

        iter_count += 1

print("Training is done.")
#Save the Model
print("Saving Model ...")

#Save the Model
torch.save(encoder,os.path.join(trained_model_output_path,"chatbot_encoder.pt"))
torch.save(decoder,os.path.join(trained_model_output_path,"chatbot_decoder.pt"))
'''