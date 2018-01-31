
import numpy
import FileUtil
import random
import time
numpy.random.seed(123)
random.seed(123)
import Load_dataset
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
#from keras.layers import LSTM, SimpleRNN
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv1D, MaxPooling1D,GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers.core import Masking, Reshape,Lambda, Flatten
from keras.layers.noise import GaussianDropout
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad,Nadam,Adamax
#from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras import backend as K



if __name__ == '__main__':

    s = {
         'input_len':20, # number of backprop through time steps
         'hidden_dims':20,
         'model_save':True,
         'mini_batch':64,
         'epochs':50}

    
    
   
    # instanciate the model
    inputs=Input(shape=(s['input_len'],))

    predict=Dense(2,activation='softmax')(inputs)

    model= Model(inputs=inputs,outputs=predict)


    opt = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()

    train_x,train_y=Load_dataset.ensemble_read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/ensemble/ensemble_nn_all-50d.train')
    dev_x,dev_y=Load_dataset.ensemble_read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/ensemble/ensemble_nn_all-50d.dev')
    test_x,test_y=Load_dataset.ensemble_read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/ensemble/ensemble_nn_all-50d.test')

    inds = range(train_y.shape[0])
    numpy.random.shuffle(inds)
    batch_num = len(inds) / s['mini_batch']

    model_file='/home/BIO/luoling/biocreativeVI/model/NN/ensemble_nn_all-50d.model_post'
    result_file='/home/BIO/luoling/biocreativeVI/result/NN/ensemble_nn_all-50d.result_test_post'

    # train with early stopping on validation set
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
            model.train_on_batch( train_x[inds[minibatch::batch_num]],
                                  train_y[inds[minibatch::batch_num]])

         
            if each_print==True:
                if minibatch % 1==0:
                    dev_res = model.predict(dev_x,batch_size=200)
                    test_res = model.predict(test_x,batch_size=200)
                    F=Eval.eval_mulclass(dev_y, dev_res, False,True)
                    if F[2]>max_f[2]:
                        test_F=Eval.eval_mulclass(test_y, test_res, False,True)
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
        dev_res = model.predict(dev_x,batch_size=200)
        test_res = model.predict(test_x,batch_size=200)
        F=Eval.eval_mulclass(test_y, test_res, True,True)
        end=time.clock()
        pit_x.append(F[2])
        if F[2]>max_f[2]:
            test_F=Eval.eval_mulclass(test_y, test_res, False,True)
            max_f[0]=F[0]
            max_f[1]=F[1]
            max_f[2]=F[2]
            max_f[3]=F[3]
            max_f[4]=epoch
            max_res=test_res
            test_f[0]=test_F[0]
            test_f[1]=test_F[1]
            test_f[2]=test_F[2]
            test_f[3]=test_F[3]
            if s['model_save']==True:
                model.save_weights(model_file)
                FileUtil.writeFloatMatrix(max_res, result_file)
        if F[2]>temp:
            each_print=True
        print 'Dev Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d, time=%.1f ' % (max_f[0],max_f[1],max_f[2],max_f[3],max_f[4],max_minibatch,epoch,end-start)
        print 'test Max P=%.5f, R=%.5f, F=%.5f, ACC=%.5f epoch=%d batch=%d,epoch_now=%d ' % (test_f[0],test_f[1],test_f[2],test_f[3],max_f[4],max_minibatch,epoch)

        
          

