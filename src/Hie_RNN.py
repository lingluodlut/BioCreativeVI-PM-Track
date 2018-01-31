
import numpy
import FileUtil
import random
import time
import Load_dataset 
numpy.random.seed(123)
random.seed(123)

import time
import sys
import subprocess
import os
from Represent_luo import RepresentationLayer
import draw_design
import Eval

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Input
from keras.layers.recurrent import LSTM, SimpleRNN,GRU
from keras.layers.merge import Concatenate,Add
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D,GlobalAveragePooling1D,LocallyConnected1D
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad,Nadam,Adamax
#from keras.layers.normalization import BatchNormalization
from keras import initializers
from AttentionLayer import Attention
from keras import backend as K



if __name__ == '__main__':

    s = {
         'pos_dim':5,
         'pos_fea':False,
         'ner_dim':5,
         'ner_fea':False,
         'sent_max_num':15, # number of backprop through time steps
         'word_max_num':30,
         'ppi_max_len':400,
         'ppi_pre':True,
         'ppilayer_train':True,
         'model_save':False,
         'hidden_dims':50,
         'mini_batch':32,
         'epochs':20}

    print s
    model_file='/home/BIO/luoling/biocreativeVI/model/ppiac_premodel/bilstm-att-token-1l-50d.ppiac3_model'
    folder = '/home/BIO/luoling/biocreativeVI/model/word_embedding_model/fastText/'
    rep1 = RepresentationLayer(folder + 'BioCreativevi_200.vec',POS_vocab_file='./fea_vocab/POS.vocab',NER_vocab_file='./fea_vocab/NER.vocab',vec_size=200, frequency=800000)
    
    
   
    # instanciate the model
    sent_input=Input(shape=(s['word_max_num'],), dtype='int32', name='sent_input')

    word_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], \
                        weights=[rep1.vec_table], input_length=s['word_max_num'])
    sent_vec=word_emb(sent_input)
    
    POS_fea=Input(shape=(s['word_max_num'],), dtype='int32', name='POS_fea')
    NER_fea=Input(shape=(s['word_max_num'],), dtype='int32', name='NER_fea')
    
    #POS_fea=Input(shape=(s['max_len'],), dtype='int32', name='POS_fea')
    pos_emb=Embedding(len(rep1.pos_fea_index),s['pos_dim'] , input_length=s['word_max_num'])
    pos_vec=pos_emb(POS_fea)
    pos_vec=Dropout(0.2)(pos_vec)
    pos_lstm=LSTM(5,implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(pos_vec)
    pos_pool=GlobalMaxPooling1D()(pos_lstm)
    posEncoder=Model([POS_fea],pos_pool)
        
    #NER_fea=Input(shape=(s['max_len'],), dtype='int32', name='NER_fea')
    ner_emb=Embedding(len(rep1.ner_fea_index),s['ner_dim'] , input_length=s['word_max_num'])
    ner_vec=ner_emb(NER_fea)
    ner_vec=Dropout(0.2)(ner_vec)
    ner_lstm=LSTM(5,implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(ner_vec)
    ner_pool=GlobalMaxPooling1D()(ner_lstm)
    nerEncoder=Model([NER_fea],ner_pool)
    #sent_vec = Concatenate()([sent_vec, pos_vec, ner_vec])


    sent_vec_drop=Dropout(0.2)(sent_vec)
#    model.add(BatchNormalization())
    sent_lstm=LSTM(50,implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(sent_vec_drop)
#    sent_tanh=TimeDistributed(Dense(50, activation = "tanh"))(sent_lstm)    
#    sent_pool=Attention()(sent_lstm)
    sent_pool=GlobalMaxPooling1D()(sent_lstm)
    #sentEncoder=Model([sent_input,POS_fea,NER_fea],sent_pool)
    sentEncoder=Model([sent_input],sent_pool)

    doc_input=Input(shape=(s['sent_max_num'],s['word_max_num']), dtype='int32', name='doc_input')
    doc_input_pos=Input(shape=(s['sent_max_num'],s['word_max_num']), dtype='int32', name='doc_input_pos')
    doc_input_ner=Input(shape=(s['sent_max_num'],s['word_max_num']), dtype='int32', name='doc_input_ner')
    
    doc_encoder = TimeDistributed(sentEncoder)(doc_input)
    doc_encoder_pos = TimeDistributed(posEncoder)(doc_input_pos)
    doc_encoder_ner = TimeDistributed(nerEncoder)(doc_input_ner)
    if s['pos_fea']==True and s['ner_fea']==False:
        doc_encoder=Concatenate()([doc_encoder,doc_encoder_pos])
    elif s['pos_fea']==False and s['ner_fea']==True:
        doc_encoder=Concatenate()([doc_encoder,doc_encoder_ner])
    elif s['pos_fea']==True and s['ner_fea']==True:
        doc_encoder=Concatenate()([doc_encoder,doc_encoder_pos,doc_encoder_ner])
    
    doc_lstm_left=LSTM(100,implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=False)(doc_encoder)
    doc_lstm_right=LSTM(100,implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=False,go_backwards=True)(doc_encoder)
    bilstm=Concatenate()([doc_lstm_left,doc_lstm_right])
    word=Input(shape=(s['ppi_max_len'],), dtype='int32', name='word')
    if s['ppi_pre']==True:
        #word=Input(shape=(s['ppi_max_len'],), dtype='int32', name='word')
        #ppi_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], weights=[rep1.vec_table], input_length=s['ppi_max_len'],name='emb',trainable=s['ppilayer_train'])
        ppi_emb=Embedding(rep1.vec_table.shape[0], 50, input_length=s['ppi_max_len'],name='emb',trainable=s['ppilayer_train'])
        ppi_vec=ppi_emb(word)
        #ppi_vec=Dropout(0.2)(ppi_vec)
	ppi_lstm=Bidirectional(LSTM(50,implementation=2,dropout=0.2,recurrent_dropout=0.2,return_sequences=True,trainable=s['ppilayer_train']),name='ppi_bilstm')(ppi_vec)
	ppi_att=Attention(name='ppi_att',trainable=s['ppilayer_train'])(ppi_lstm)
	ppi_fc=Dense(100,activation='relu',name='ppi_dense1',trainable=s['ppilayer_train'])(ppi_att)
	ppi_fc=Dense(50,activation='relu',name='ppi_dense2',trainable=s['ppilayer_train'])(ppi_fc)
        bilstm=Concatenate()([bilstm,ppi_fc])   
    predict=Dense(2,activation='softmax',name='lastpre')(bilstm)

    model= Model(inputs=[doc_input,doc_input_pos,doc_input_ner,word],outputs=predict)
    

    opt = RMSprop(0.001)

    model.load_weights(model_file, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()


    devfile='/home/BIO/luoling/biocreativeVI/data/ppiac/PMtask_Triage_DevSet_ir_scon.hiernn_PPIAC_fea'
    fin_dev=open(devfile,'r')
    dev=[]
    sent=[]
    for line in fin_dev:
        line=line.strip()
        if line!='':
            sent.append(line)
        else:
            dev.append(sent)
            sent=[]
    #dev_x1,dev_y1=rep1.represent_instances_3D(dev, s['sent_max_num'],s['word_max_num'])
    dev_x1,dev_pos1,dev_ner1,dev_y1=rep1.represent_instances_3D_fea(dev, s['sent_max_num'],s['word_max_num'])
    fin_dev.close()
    
    trainfile='/home/BIO/luoling/biocreativeVI/data/ppiac/PMtask_Triage_TrainingSet_ir_scon.hiernn_PPIAC_fea'
    fin_train=open(trainfile,'r')
    train=[]
    sent=[]
    for line in fin_train:
        line=line.strip()
        if line!='':
            sent.append(line)
        else:
            train.append(sent)
            sent=[]
    #train_x1,train_y1=rep1.represent_instances_3D(train, s['sent_max_num'],s['word_max_num'])
    train_x1,train_pos1,train_ner1,train_y1=rep1.represent_instances_3D_fea(train, s['sent_max_num'],s['word_max_num'])
    fin_train.close()    

    testfile='/home/BIO/luoling/biocreativeVI/data/ppiac/PMtask_TestSet_gold.hiernn_PPIAC_fea'
    fin_test=open(testfile,'r')
    test=[]
    sent=[]
    for line in fin_test:
        line=line.strip()
        if line!='':
            sent.append(line)
        else:
            test.append(sent)
            sent=[]
    #test_x1,test_y1=rep1.represent_instances_3D(test, s['sent_max_num'],s['word_max_num'])
    test_x1,test_pos1,test_ner1,test_y1=rep1.represent_instances_3D_fea(test, s['sent_max_num'],s['word_max_num'])
    fin_test.close()
#    print train_x1


    train_token_text,train_pos_text,train_ner_text=Load_dataset.read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/fea-add/PMtask_Triage_TrainingSet_ir_scon.conll.PPIAC')
    dev_token_text,dev_pos_text,dev_ner_text=Load_dataset.read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/fea-add/PMtask_Triage_DevSet_ir_scon.conll.PPIAC')
    test_token_text,test_pos_text,test_ner_text=Load_dataset.read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/test/PMtask_TestSet_gold.fea_conll.PPIAC')
#    print test_token_text[0],test_pos_text[0],test_ner_text[0]
#    train1 = [line.strip() for line in open(folder + 'corpus/testfile1.txt')
   
    
    test_token,test_pos,test_ner, test_y = rep1.represent_instances_fea(test_token_text,test_pos_text,test_ner_text, max_len=s['ppi_max_len'])
    dev_token, dev_pos, dev_ner, dev_y = rep1.represent_instances_fea(dev_token_text, dev_pos_text, dev_ner_text, max_len=s['ppi_max_len'])
    train_token,train_pos, train_ner,train_y = rep1.represent_instances_fea(train_token_text,train_pos_text,train_ner_text, max_len=s['ppi_max_len'])


    inds = range(train_y1.shape[0])
    numpy.random.shuffle(inds)
    batch_num = len(inds) / s['mini_batch']

    # train with early stopping on validation set
    model_file='/home/BIO/luoling/biocreativeVI/model/NN/hiernn-token-ppipre-100d-tune.model_post'
    result_file='/home/BIO/luoling/biocreativeVI/result/NN/hiernn-token-ppipre-100d-tune.result_test_post'

    best_f1 = -numpy.inf    
    iter = 0
    max_f=[0,0,0,0,-1]
    test_f=[0,0,0,0,-1]
    max_res=0
    each_print=False
    max_minibatch=0
    temp=0.6
    pit_x=[]
#    a=keras.backend.variable(np.random.uniform(-5,5,[5,5]))
#    print a.get_value()
    for epoch in xrange(s['epochs']):
#        start=time.clock()
        for minibatch in range(batch_num):
            
            model.train_on_batch( [train_x1[inds[minibatch::batch_num]],
                                   train_pos1[inds[minibatch::batch_num]],
                                   train_ner1[inds[minibatch::batch_num]],
                                   train_token[inds[minibatch::batch_num]]
                                  ],
                                  train_y1[inds[minibatch::batch_num]])

        
        # evaluation // back into the real world : idx -> words
            if each_print==True:
                if minibatch % 1==0: 
                    dev_res = model.predict([dev_x1,dev_pos1,dev_ner1,dev_token],batch_size=200)
                    test_res = model.predict([test_x1,test_pos1,test_ner1,test_token],batch_size=200)
                    F=Eval.eval_mulclass(dev_y1, dev_res, False,True)
                    if F[2]>max_f[2]:
                        test_F=Eval.eval_mulclass(test_y1, test_res, False,True)
                        max_f[0]=F[0]
                        max_f[1]=F[1]
                        max_f[2]=F[2]
                        max_f[3]=F[3]
                        max_f[4]=epoch
                        test_f[0]=test_F[0]
                        test_f[1]=test_F[1]
                        test_f[2]=test_F[2]
                        test_f[3]=test_F[3]
                        max_res=test_res
                        max_minibatch=minibatch
                        if s['model_save']==True:
                            model.save_weights(model_file)
                            FileUtil.writeFloatMatrix(max_res, result_file)
#        end=time.clock()
        start=time.clock()
        dev_res = model.predict([dev_x1,dev_pos1,dev_ner1,dev_token],batch_size=200)
        test_res = model.predict([test_x1,test_pos1,test_ner1,test_token],batch_size=200)
        F=Eval.eval_mulclass(dev_y1, dev_res, True,True)
        #pit_x.append(F[2])
        end=time.clock()
        if F[2]>max_f[2]:
            test_F=Eval.eval_mulclass(test_y1, test_res, False,True)
            max_f[0]=F[0]
            max_f[1]=F[1]
            max_f[2]=F[2]
            max_f[3]=F[3]
            max_f[4]=epoch
            test_f[0]=test_F[0]
            test_f[1]=test_F[1]
            test_f[2]=test_F[2]
            test_f[3]=test_F[3]
            max_res=test_res
            if s['model_save']==True:
                model.save_weights(model_file)
                FileUtil.writeFloatMatrix(max_res, result_file)
        if F[2]>temp:
            each_print=True
        print 'Dev Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d, ' % (max_f[0],max_f[1],max_f[2],max_f[3],max_f[4],max_minibatch,epoch)
        print 'Test Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d, ' % (test_f[0],test_f[1],test_f[2],test_f[3],max_f[4],max_minibatch,epoch)

          

