
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
from keras.layers import LSTM, SimpleRNN,GRU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.layers.noise import GaussianDropout
from keras.layers.merge import Concatenate,Maximum,Average
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad,Nadam,Adamax
#from keras.layers.normalization import BatchNormalization
from AttentionLayer import Attention
from keras import initializers
from keras import backend as K
        
       

if __name__ == '__main__':

    s = {
         'pos_dim':5,
         'pos_fea':False,
         'ner_dim':5,
         'ner_fea':False, 
         'max_len':400, # number of backprop through time steps
         'lstm_h':50,
         'hidden_dims':30,
	 'ppi_pre':False,
	 'ppilayer_train':True,
         'mini_batch':32,
         'epochs':30,
         'model_save':False,
         'model':'LSTM'} #LSTM,CNN,LSTM-CNN,RCNN

    print s
    model_file='/home/BIO/luoling/biocreativeVI/model/ppiac_premodel/lstm-token-1l-50d.ppiac3_model'
    folder1 = '/home/BIO/luoling/biocreativeVI/model/word_embedding_model/fastText/'
    rep1 = RepresentationLayer(folder1 + 'BioCreativevi.vec',POS_vocab_file='./fea_vocab/POS.vocab',NER_vocab_file='./fea_vocab/NER.vocab',vec_size=50, frequency=800000)
    print len(rep1.ner_fea_index), len(rep1.pos_fea_index)
    print rep1.vec_table.shape[0],rep1.vec_table.shape[1]
   
    
    word=Input(shape=(s['max_len'],), dtype='int32', name='word')
    word_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], weights=[rep1.vec_table], input_length=s['max_len'],name='emb')
    word_vec=word_emb(word)
    
    if s['pos_fea']==True and s['ner_fea']==False:
        POS_fea=Input(shape=(s['max_len'],), dtype='int32', name='POS_fea')
        pos_emb=Embedding(len(rep1.pos_fea_index),s['pos_dim'] , input_length=s['max_len'])
        pos_vec=pos_emb(POS_fea)
        word_vec = Concatenate()([word_vec, pos_vec])

    elif s['pos_fea']==False and s['ner_fea']==True:
        NER_fea=Input(shape=(s['max_len'],), dtype='int32', name='NER_fea')
        ner_emb=Embedding(len(rep1.ner_fea_index),s['ner_dim'] , input_length=s['max_len'])
        ner_vec=ner_emb(NER_fea)
        word_vec = Concatenate()([word_vec, ner_vec])

    elif s['pos_fea']==True and s['ner_fea']==True:
        POS_fea=Input(shape=(s['max_len'],), dtype='int32', name='POS_fea')
        pos_emb=Embedding(len(rep1.pos_fea_index),s['pos_dim'] , input_length=s['max_len'])
        pos_vec=pos_emb(POS_fea)
       
        NER_fea=Input(shape=(s['max_len'],), dtype='int32', name='NER_fea')
        ner_emb=Embedding(len(rep1.ner_fea_index),s['ner_dim'] , input_length=s['max_len'])
        ner_vec=ner_emb(NER_fea)
        word_vec = Concatenate()([word_vec, pos_vec, ner_vec])

    input_vec=Dropout(0.2)(word_vec)
#    input_vec=word_vec
 
    if s['model']=='LSTM':
        lstm=LSTM(s['lstm_h'],implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=True)(input_vec)
        tanhl=TimeDistributed(Dense(20, activation = "tanh"))(lstm)

        pool=GlobalMaxPooling1D()(lstm)
        
        

    elif s['model']=='CNN':
      
        cnn=Conv1D(150, 3,padding='same',activation='relu')(input_vec)
        cnn=Conv1D(150, 3,padding='same',activation='relu')(cnn)

        pool=GlobalMaxPooling1D()(cnn)

        pool=Dense(s['hidden_dims'],activation='relu',name='cnn_dense')(pool)
        pool=Dropout(0.2)(pool)
    
    
    elif s['model']=='LSTM-CNN':
        lstm=Bidirectional(LSTM(s['lstm_h'],implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(input_vec)
        tanhl=TimeDistributed(Dense(20, activation = "tanh"))(lstm)
        cnn=cnn=Conv1D(60, 3,padding='same',activation='tanh')(tanhl)
        pool=GlobalMaxPooling1D()(cnn)
        pool=Dense(s['hidden_dims'],activation='tanh',name='lc_dense')(pool)
        pool=Dropout(0.2)(pool)

    elif s['model']=='RCNN':
        l_context=Input(shape=(s['max_len'],), dtype='int32', name='l_context')
        r_context=Input(shape=(s['max_len'],), dtype='int32', name='r_context')
        l_vec=word_emb(l_context)
        r_vec=word_emb(r_context)
        l_vec_input=Dropout(0.2)(l_vec)
        r_vec_input=Dropout(0.2)(r_vec)
        forward=LSTM(50,implementation=2,dropout=0, recurrent_dropout=0,return_sequences=True)(l_vec_input)
        backward=LSTM(50,implementation=2,dropout=0, recurrent_dropout=0,return_sequences=True,go_backwards=True)(r_vec_input)
        together=Concatenate()([forward,input_vec,backward])
        tanhl=TimeDistributed(Dense(150, activation = "tanh"))(together)
        pool=GlobalMaxPooling1D()(tanhl)

    if s['ppi_pre']==True:
        ppi_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], weights=[rep1.vec_table], input_length=s['max_len'],name='emb',trainable=s['ppilayer_train'])
        ppi_vec=ppi_emb(word)
	ppi_lstm=Bidirectional(LSTM(50,implementation=2,dropout=0.2,recurrent_dropout=0.2,return_sequences=True,trainable=s['ppilayer_train']),name='ppi_bilstm')(ppi_vec)
	ppi_att=Attention(name='ppi_att',trainable=s['ppilayer_train'])(ppi_lstm)
	ppi_fc=Dense(100,activation='relu',name='ppi_dense1',trainable=s['ppilayer_train'])(ppi_att)
	ppi_fc=Dense(50,activation='relu',name='ppi_dense2',trainable=s['ppilayer_train'])(ppi_fc)
        pool=Concatenate()([pool,ppi_fc])

    predict=Dense(2,activation='softmax',name='ppi_pre')(pool)
    if s['pos_fea']==True and s['ner_fea']==False:
        if s['model']=='RCNN':
            model= Model(inputs=[word,l_context,r_context,POS_fea],outputs=predict)
        else:
            model= Model(inputs=[word,POS_fea],outputs=predict)
    elif s['pos_fea']==False and s['ner_fea']==True:
        if s['model']=='RCNN':
            model= Model(inputs=[word,l_context,r_context,NER_fea],outputs=predict)
        else:
            model= Model(inputs=[word,NER_fea],outputs=predict)
    elif s['pos_fea']==True and s['ner_fea']==True:
        if s['model']=='RCNN':
            model= Model(inputs=[word,l_context,r_context,POS_fea,NER_fea],outputs=predict)
        else:
            model= Model(inputs=[word,POS_fea,NER_fea],outputs=predict)
    elif s['pos_fea']==False and s['ner_fea']==False: 
        if s['model']=='RCNN':
            model= Model(inputs=[word,l_context,r_context],outputs=predict)
        else:
            model= Model(inputs=word,outputs=predict)

    if s['model'] in ['LSTM','LSTM-CNN','RCNN']:
        opt = RMSprop(lr=0.001)
    elif s['model'] in ['CNN']:
        opt = Adadelta()

    model.load_weights(model_file, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()


    train_token_text,train_pos_text,train_ner_text=Load_dataset.read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/fea-add/PMtask_Triage_TrainingSet_ir_scon.conll.PPIAC')
    dev_token_text,dev_pos_text,dev_ner_text=Load_dataset.read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/fea-add/PMtask_Triage_DevSet_ir_scon.conll.PPIAC')
    test_token_text,test_pos_text,test_ner_text=Load_dataset.read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/test/PMtask_TestSet_gold.fea_conll.PPIAC')

   
    train_x1,l_train_x1,r_train_x1, train_y1 = rep1.represent_instances_rcnn(train_token_text, max_len=s['max_len'])
    dev_x1,l_dev_x1,r_dev_x1, dev_y1 = rep1.represent_instances_rcnn(dev_token_text, max_len=s['max_len'])
    test_x1,l_test_x1,r_test_x1, test_y1 = rep1.represent_instances_rcnn(test_token_text, max_len=s['max_len'])
    
    test_token,test_pos,test_ner, test_y = rep1.represent_instances_fea(test_token_text,test_pos_text,test_ner_text, max_len=s['max_len'])
    dev_token, dev_pos, dev_ner, dev_y = rep1.represent_instances_fea(dev_token_text, dev_pos_text, dev_ner_text, max_len=s['max_len'])
    train_token,train_pos, train_ner,train_y = rep1.represent_instances_fea(train_token_text,train_pos_text,train_ner_text, max_len=s['max_len'])

    inds = range(train_y.shape[0])
    numpy.random.shuffle(inds)
    batch_num = len(inds) / s['mini_batch']

    # train with early stopping on validation set

    result_file='/home/BIO/luoling/biocreativeVI/result/NN/cnn-token_fea-ppipre-50d-tune.result_test_post2'
    model_file='/home/BIO/luoling/biocreativeVI/model/NN/cnn-token_fea-ppipre-50d-tune.model_post2'

    best_f1 = -numpy.inf    
    iter = 0
    max_f=[0,0,0,0,-1]
    test_f=[0,0,0,0,-1]
    max_res=0
    each_print=False
    max_minibatch=0
    temp=0.6
    pit_x=[]

    for epoch in xrange(s['epochs']):
#        start=time.clock()
        for minibatch in range(batch_num):
            if s['pos_fea']==True and s['ner_fea']==False:
                if s['model']=='RCNN':
                    model.train_on_batch( [train_x1[inds[minibatch::batch_num]],
                                   l_train_x1[inds[minibatch::batch_num]],
                                   r_train_x1[inds[minibatch::batch_num]],
                                   train_pos[inds[minibatch::batch_num]]
                                  ],
                                  train_y1[inds[minibatch::batch_num]])
                else:
                    model.train_on_batch( [train_token[inds[minibatch::batch_num]],train_pos[inds[minibatch::batch_num]]],
                                      train_y[inds[minibatch::batch_num]])
            
            elif s['pos_fea']==False and s['ner_fea']==True:
                if s['model']=='RCNN':
                    model.train_on_batch( [train_x1[inds[minibatch::batch_num]],
                                   l_train_x1[inds[minibatch::batch_num]],
                                   r_train_x1[inds[minibatch::batch_num]],
                                   train_ner[inds[minibatch::batch_num]]
                                  ],
                                  train_y1[inds[minibatch::batch_num]])
                else:
                    model.train_on_batch( [train_token[inds[minibatch::batch_num]],train_ner[inds[minibatch::batch_num]]],
                                      train_y[inds[minibatch::batch_num]])
            elif s['pos_fea']==True and s['ner_fea']==True:
                if s['model']=='RCNN':
                    model.train_on_batch( [train_x1[inds[minibatch::batch_num]],
                                   l_train_x1[inds[minibatch::batch_num]],
                                   r_train_x1[inds[minibatch::batch_num]],
                                   train_pos[inds[minibatch::batch_num]],
                                   train_ner[inds[minibatch::batch_num]]
                                  ],
                                  train_y1[inds[minibatch::batch_num]])
                else:
                    model.train_on_batch( [train_token[inds[minibatch::batch_num]],train_pos[inds[minibatch::batch_num]],train_ner[inds[minibatch::batch_num]]],
                                      train_y[inds[minibatch::batch_num]])

            elif s['pos_fea']==False and s['ner_fea']==False:
                if s['model']=='RCNN':
                    model.train_on_batch( [train_x1[inds[minibatch::batch_num]],
                                   l_train_x1[inds[minibatch::batch_num]],
                                   r_train_x1[inds[minibatch::batch_num]]
                                  ],
                                  train_y1[inds[minibatch::batch_num]])
                else:
                    model.train_on_batch( train_token[inds[minibatch::batch_num]],
                                      train_y[inds[minibatch::batch_num]])

                  # evaluation // back into the real world : idx -> words
            if each_print==True:
                if minibatch % 1==0:
                    if s['pos_fea']==True and s['ner_fea']==False:
                        if s['model']=='RCNN':
                            dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1,dev_pos],batch_size=200)
                            test_res = model.predict([test_x1,l_test_x1,r_test_x1,test_pos],batch_size=200)
                        else:
                            dev_res = model.predict([dev_token,dev_pos],batch_size=200)
                            test_res = model.predict([test_token,test_pos],batch_size=200)
                    elif s['pos_fea']==False and s['ner_fea']==True:
                        if s['model']=='RCNN':
                            dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1,dev_ner],batch_size=200)
                            test_res = model.predict([test_x1,l_test_x1,r_test_x1,test_ner],batch_size=200)
                        else:
                            dev_res = model.predict([dev_token,dev_ner],batch_size=200)
                            test_res = model.predict([test_token,test_ner],batch_size=200)
                    elif s['pos_fea']==True and s['ner_fea']==True:
                        if s['model']=='RCNN':
                            dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1,dev_pos,dev_ner],batch_size=200)
                            test_res = model.predict([test_x1,l_test_x1,r_test_x1,test_pos,test_ner],batch_size=200)
                        else:
                            dev_res = model.predict([dev_token,dev_pos,dev_ner],batch_size=200)
                            test_res = model.predict([test_token,test_pos,test_ner],batch_size=200)
                    elif s['pos_fea']==False and s['ner_fea']==False:
                        if s['model']=='RCNN':
                            dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1],batch_size=200)
                            test_res = model.predict([test_x1,l_test_x1,r_test_x1],batch_size=200)
                        else:
                            dev_res = model.predict(dev_token,batch_size=200)
                            test_res = model.predict(test_token,batch_size=200)
                    F=Eval.eval_mulclass(dev_y, dev_res, False,True)
                    if F[2]>max_f[2]:
                        test_F=Eval.eval_mulclass(test_y, test_res, False, True)
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

        start=time.clock()
        if s['pos_fea']==True and s['ner_fea']==False:
            if s['model']=='RCNN':
                dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1,dev_pos],batch_size=200)
                test_res = model.predict([test_x1,l_test_x1,r_test_x1,test_pos],batch_size=200)
            else:
                dev_res = model.predict([dev_token,dev_pos],batch_size=200)
                test_res = model.predict([test_token,test_pos],batch_size=200)

        elif s['pos_fea']==False and s['ner_fea']==True:
            if s['model']=='RCNN':
                dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1,dev_ner],batch_size=200)
                test_res = model.predict([test_x1,l_test_x1,r_test_x1,test_ner],batch_size=200)
            else:
                dev_res = model.predict([dev_token,dev_ner],batch_size=200)
                test_res = model.predict([test_token,test_ner],batch_size=200)
        elif s['pos_fea']==True and s['ner_fea']==True:
            if s['model']=='RCNN':
                dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1,dev_pos,dev_ner],batch_size=200)
                test_res = model.predict([test_x1,l_test_x1,r_test_x1,test_pos,test_ner],batch_size=200)
            else:
                dev_res = model.predict([dev_token,dev_pos,dev_ner],batch_size=200)
                test_res = model.predict([test_token,test_pos,test_ner],batch_size=200)
        elif s['pos_fea']==False and s['ner_fea']==False:
            if s['model']=='RCNN':
                dev_res = model.predict([dev_x1,l_dev_x1,r_dev_x1],batch_size=200)
                test_res = model.predict([test_x1,l_test_x1,r_test_x1],batch_size=200)
            else:
                dev_res = model.predict(dev_token,batch_size=200)
                test_res = model.predict(test_token,batch_size=200)
        F=Eval.eval_mulclass(dev_y, dev_res, True,True)
        end=time.clock()
        pit_x.append(F[2])
        if F[2]>max_f[2]:
            test_F=Eval.eval_mulclass(test_y, test_res, False, True)
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
        print 'Test P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d, ' % (test_f[0],test_f[1],test_f[2],test_f[3],max_f[4],max_minibatch,epoch)

    print model_file,'done'
        
          

