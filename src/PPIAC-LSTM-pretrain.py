
import FileUtil
import numpy
import time
import sys
import subprocess
import os
import Load_dataset
import random
from Represent_luo import RepresentationLayer
from AttentionLayer import Attention
numpy.random.seed(123)
random.seed(123)

import Eval

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Lambda
from keras.layers import Embedding,Input
from keras.layers import LSTM, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad, Nadam
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate,Average
from keras import backend as K

#def max_1d(X):
#    return K.max(X, axis=1)


if __name__ == '__main__':

    s = {
         'max_len_token':400, 
         'emb_dimension':50, # dimension of word embedding
         'hidden_dims':50,
         'mini_batch':32,
         'epochs':15,
         'each_print':True,
         'model_save':False,
        }

    print s
    folder = '/home/BIO/luoling/PPIAC_CNN/'
    rep1 = RepresentationLayer(
            '/home/BIO/luoling/biocreativeVI/model/word_embedding_model/fastText/BioCreativevi.vec,\
            vec_size=50, frequency=800000)

 

#chanel1    
    chanel1_input=Input(shape=(s['max_len_token'],),dtype='int32')
    chanel1_emb=Embedding(rep1.vec_table.shape[0], rep1.vec_table.shape[1], \
                          weights=[rep1.vec_table], input_length=s['max_len_token'],name='emb')
    chanel1_vec=chanel1_emb(chanel1_input)

    chanel1_vec=Dropout(0.2)(chanel1_vec)

    chanel1_lstm1=Bidirectional(LSTM(50, implementation=2,dropout=0.2, recurrent_dropout=0.2,return_sequences=True),name='ppi_bilstm')(chanel1_vec)

    chanel1_pool=Attention(name='ppi_att')(chanel1_lstm1)

    
    concat_fc=Dense(100,activation='relu',name='ppi_dense1')(chanel1_pool)
    concat_fc=Dense(50,activation='relu',name='ppi_dense2')(concat_fc)
    concat_fc=Dropout(0.2)(concat_fc)
    
    predict=Dense(2,activation='softmax',name='ppi_pre')(concat_fc)

    model= Model(inputs=chanel1_input,outputs=predict)

    opt = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()

    #data set

        test1 = [line.strip() for line in open(folder + 'corpus/bc3_ppi_test.token.lab')]

        train1 = [line.strip() for line in open(folder + 'corpus/bc3_dev_ppi_train.token.lab')]

        #10% to dev, remaining to train
        dev_num=int(len(train1)*0.1)
        bc3_dev=train1[0:4000]
        random.shuffle(bc3_dev)
        new_dev=bc3_dev[0:dev_num]
        train1=bc3_dev[dev_num:]+train1[4000:]
        random.shuffle(train1)
        dev1=new_dev


    train_x1, train_y1 = rep1.represent_instances(train1, max_len=s['max_len_token'])
    dev_x1, dev_y1 = rep1.represent_instances(dev1,max_len=s['max_len_token'])
    test_x1, test_y1 = rep1.represent_instances(test1, max_len=s['max_len_token'])


    inds = range(train_y1.shape[0])
    numpy.random.shuffle(inds)
    batch_num = len(inds) / s['mini_batch']

    # train with early stopping on validation set
    best_f1 = -numpy.inf    
    iter = 0
    max_f=[0,0,0,0,-1]
    last_test_f=[0,0,0,0,-1]
    max_res=0
    temp=0.93
    each_print=s['each_print']
    model_save=s['model_save']
    max_minibatch=0

    result_file='/home/BIO/luoling/biocreativeVI/model/ppiac_premodel/lstm-token-1l-50d.ppiac3_result'
    model_file='/home/BIO/luoling/biocreativeVI/model/ppiac_premodel/lstm-token-1l-50d.ppiac3_model'

    for epoch in xrange(s['epochs']):
        #numpy.random.shuffle(inds)
        for minibatch in range(batch_num):
            model.train_on_batch( train_x1[inds[minibatch::batch_num]],
                                  train_y1[inds[minibatch::batch_num]])
        
            if each_print==True:
                if minibatch % 1 ==0:
                    dev_res=model.predict(dev_x1,batch_size=200)
                    F=Eval.eval_mulclass(dev_y1, dev_res,False, True)

                    if F[2]>max_f[2]:
                        test_res=model.predict(test_x1,batch_size=200)
                        test_F=Eval.eval_mulclass(test_y1, test_res,False,True)
                        max_f[0]=F[0]
                        max_f[1]=F[1]
                        max_f[2]=F[2]
                        max_f[3]=F[3]
                        max_f[4]=epoch
                        last_test_f[0]=test_F[0]
                        last_test_f[1]=test_F[1]
                        last_test_f[2]=test_F[2]
                        last_test_f[3]=test_F[3]
                        max_res=test_res
                        max_minibatch=minibatch
                        if model_save==True:
                            FileUtil.writeFloatMatrix(max_res, result_file)
                            model.save_weights(model_file)

        dev_res=model.predict(dev_x1,batch_size=200)
        F=Eval.eval_mulclass(dev_y1, dev_res,True, True)
        if F[2]>max_f[2]:
            test_res=model.predict(test_x1,batch_size=200)
            test_F=Eval.eval_mulclass(test_y1, test_res,False, True)
            max_f[0]=F[0]
            max_f[1]=F[1]
            max_f[2]=F[2]
            max_f[3]=F[3]
            max_f[4]=epoch
            last_test_f[0]=test_F[0]
            last_test_f[1]=test_F[1]
            last_test_f[2]=test_F[2]
            last_test_f[3]=test_F[3]
            max_res=test_res
            if model_save==True:
                FileUtil.writeFloatMatrix(max_res, result_file)
                model.save_weights(model_file)
        if F[2]>temp:
            each_print=True
        print 'Dev Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d ' % (max_f[0],max_f[1],max_f[2],max_f[3],max_f[4],max_minibatch,epoch)
        print 'Test P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d' % (last_test_f[0],last_test_f[1],last_test_f[2],last_test_f[3],max_f[4],max_minibatch,epoch)
        print '****************************************************************************'

    print model_file,'done' 
   
        
       

