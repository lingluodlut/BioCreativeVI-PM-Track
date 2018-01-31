import os, sys
import numpy as np


class RepresentationLayer(object):
    
    
    def __init__(self, wordvec_file, POS_vocab_file=None, NER_vocab_file=None, \
                 vec_size=50, frequency=10000, scale=1):
        
        '''
        vec_size        :    the dimension size of word vector 
                             learned by word2vec tool
        
        frequency       :    the threshold for the words left according to
                             their frequency appeared in the text
                             for example, when frequency is 10000, the most
                             frequent appeared 10000 words are considered
        
        scheme          :    the NER label scheme like IOB,IOBE, IOBES
                                     
        scale           :    the scaling for the vectors' each real value
                             when the vectors are scaled up it will accelerate
                             the training process
        
        vec_talbe        :    a matrix each row stands for a vector of a word

        index_map        :    the map from word to corresponding index in vec_table
        
        pos_2_index    :    the map from a POS to corresponding vector's index
        
        chunk_2_index    : the map from a Chunk to corresponding vector's index
        
        shape_2_index    : the map from a word's shape to corresponding vector's index

        orthograph_2_index    : the map from a word's orthograph to corresponding vector's index

        affix_2_index    : the map from a word's affix to corresponding vector's index

        terminology_2_index    : the map from a word's terminology to corresponding vector's index

        length_2_index    : the map from a word's length to corresponding vector's index

        semantic_mat    :    the matrix contains disease semantic information
        
        '''
        self.frequency = frequency
        self.vec_size = vec_size
        self.scale = scale
        
        self.vec_table = np.zeros((self.frequency + 1, self.vec_size))
        self.word_2_index = {}
        self.load_wordvecs(wordvec_file)
        
        self.pos_fea_index={}
        if POS_vocab_file!=None:
            self.load_fea_vocab(POS_vocab_file,self.pos_fea_index)

        self.ner_fea_index={}
        if NER_vocab_file!=None:
            self.load_fea_vocab(NER_vocab_file,self.ner_fea_index)

    def load_fea_vocab(self,fea_file,fea_index):
        fin=open(fea_file)
        i=0
        for line in fin:
            fea_index[line.strip()]=i
            i+=1
        fin.close()
    
    def load_wordvecs(self, wordvec_file):
        
        file = open(wordvec_file)
        first_line = file.readline()
        word_count = int(first_line.split()[0])
        
        row = 0
        for line in file:
            if row < self.frequency:
                line_split = line[:-1].split()
                self.word_2_index[line_split[0]] = row
                for col in xrange(self.vec_size):
                    self.vec_table[row][col] = float(line_split[col + 1])
                row += 1
            else:
                break
        
        self.word_2_index['sparse_vectors'] = row
        sparse_vectors = np.zeros(self.vec_size)

        if word_count > self.frequency:
            for line in file:
                line_split = line[:-1].split()[1:]
                for i in xrange(self.vec_size):
                    sparse_vectors[i] += float(line_split[i])
            sparse_vectors /= (word_count - self.frequency)
        
        for col in xrange(self.vec_size):
            self.vec_table[row][col] = sparse_vectors[col]
        
     
        self.vec_table *= self.scale    
        
        file.close()

    


    def word_2_vec(self, word):
        if self.word_2_index.has_key(word):
            return self.vec_table[self.word_2_index[word]]
        else:
            return self.vec_table[self.word_2_index['sparse_vectors']]


    def labelindex_2_vec(self, label_index):
        if label_index == 0:
            return [1, 0, 0.]
        elif label_index == 1:
            return [0, 1, 0.]
        elif label_index == 2:
            return [0, 0, 1.]
        else:
            print 'Unexcepted label index'
            return None

    

    def indexs_2_labels(self, indexs):
        labels = []
        
        for index in indexs:
            labels.append(self.index_2_label(index))
        
        return labels

    def get_array_of_type2(self, type):
        if type == 'false':
            return [0.,1.]
        else:
            return [1.,0.]
    

    def represent_instance(self, instance):
        
        word_indexs = []
        splited = instance.split(' ')
        for word in splited[1:]:
            if self.word_2_index.has_key(word):
                word_indexs.append(self.word_2_index[word])
            else:
                word_indexs.append(self.word_2_index['sparse_vectors'])
        
      
        type = self.get_array_of_type2(splited[0])

        
        return type, word_indexs          
    
    def represent_instance_pos(self, instance):
        
        pos_indexs = []
        splited = instance.split(' ')
        for pos in splited[1:]:
            if self.pos_fea_index.has_key(pos):
                pos_indexs.append(self.pos_fea_index[pos])
            else:
                pos_indexs.append(0)
        
        return pos_indexs

    def represent_instance_ner(self, instance):

        ner_indexs = []
        splited = instance.split(' ')
        for ner in splited[1:]:
            if self.ner_fea_index.has_key(ner):
                ner_indexs.append(self.ner_fea_index[ner])
            else:
                ner_indexs.append(0)

        return ner_indexs


    '''
        represent instances that will pass to
        the recurrent neural networks.
        the input format are the word sequence
        and corresponding label sequence
    
    '''
    def represent_instances(self, instances, max_len=None):
        labels_list = []
        words_list = []
        
        for instance in instances:
            label_array, word_indexs = self.represent_instance(instance)
            labels_list.extend(label_array)
            if len(word_indexs) > max_len:
                words_list.extend(word_indexs[0:max_len])
            else:
                for i in range(max_len-len(word_indexs)):
                    word_indexs.append(0)
                words_list.extend(word_indexs)
        
        label_mat =  np.array(labels_list)
        label_mat = label_mat.reshape((len(instances), 2))
        
        x_mat = np.array(words_list)
        x_mat = x_mat.reshape((len(instances), max_len))
        
        return x_mat, label_mat
 
    def represent_instances_fea(self, instances,pos_instances=None, ner_instances=None,max_len=None):
        labels_list = []
        words_list = []
        pos_list=[]
        ner_list=[]
        
        for instance,pos_instance,ner_instance in zip(instances,pos_instances,ner_instances):
            label_array, word_indexs = self.represent_instance(instance)
            pos_indexs=self.represent_instance_pos(pos_instance)
            ner_indexs=self.represent_instance_ner(ner_instance)
            labels_list.extend(label_array)
            if len(word_indexs) > max_len:
                words_list.extend(word_indexs[0:max_len])
                pos_list.extend(pos_indexs[0:max_len])
                ner_list.extend(ner_indexs[0:max_len])
            else:
                for i in range(max_len-len(word_indexs)):
                    word_indexs.append(0)
                    pos_indexs.append(0)
                    ner_indexs.append(0)
                words_list.extend(word_indexs)
                pos_list.extend(pos_indexs)
                ner_list.extend(ner_indexs)
        
        label_mat =  np.array(labels_list)
        label_mat = label_mat.reshape((len(instances), 2))
        
        x_mat = np.array(words_list)
        x_mat = x_mat.reshape((len(instances), max_len))
        
        pos_mat=np.array(pos_list)
        pos_mat=pos_mat.reshape((len(instances), max_len))
 
        ner_mat=np.array(ner_list)
        ner_mat=ner_mat.reshape((len(instances), max_len))
        
        return x_mat ,pos_mat, ner_mat, label_mat

    def represent_instances_fea_ner(self, instances,ner_instances=None, max_len=None):
        labels_list = []
        words_list = []
        ner_list=[]

        for instance,ner_instance in zip(instances,ner_instances):
            label_array, word_indexs = self.represent_instance(instance)
            ner_indexs=self.represent_instance_ner(ner_instance)
            labels_list.extend(label_array)
            if len(word_indexs) > max_len:
                words_list.extend(word_indexs[0:max_len])
                ner_list.extend(ner_indexs[0:max_len])
            else:
                for i in range(max_len-len(word_indexs)):
                    word_indexs.append(0)
                    ner_indexs.append(0)
                words_list.extend(word_indexs)
                ner_list.extend(ner_indexs)

        label_mat =  np.array(labels_list)
        label_mat = label_mat.reshape((len(instances), 2))

        x_mat = np.array(words_list)
        x_mat = x_mat.reshape((len(instances), max_len))

        ner_mat=np.array(ner_list)
        ner_mat=ner_mat.reshape((len(instances), max_len))

        return x_mat ,ner_mat,label_mat

    def represent_instances_rcnn(self,instances,max_len=None):
        labels_list=[]
        words_list=[]
        l_words_list=[]
        r_words_list=[]
        for instance in instances:
            label_array,word_indexs=self.represent_instance(instance)
            labels_list.extend(label_array)
            if len(word_indexs) > max_len:
                words_list.extend(word_indexs[0:max_len])
            else:
                for i in range(max_len-len(word_indexs)):
                    word_indexs.append(0)
                words_list.extend(word_indexs)
            l_words_list.extend([0])
            l_words_list.extend(word_indexs[:max_len-1])
            r_words_list.extend(word_indexs[1:max_len])
            r_words_list.extend([0]) 
        label_mat = np.array(labels_list)
        label_mat=label_mat.reshape((len(instances),2))
        x_mat=np.array(words_list)
        x_mat=x_mat.reshape((len(instances),max_len))
        x_l_mat=np.array(l_words_list)
        x_l_mat=x_l_mat.reshape((len(instances),max_len))
        x_r_mat=np.array(r_words_list)
        x_r_mat=x_r_mat.reshape((len(instances),max_len))
         
        return x_mat,x_l_mat,x_r_mat,label_mat
    
    def represent_instances_3D(self,instances,sent_max_num=None,word_max_num=None):
        data = np.zeros((len(instances), sent_max_num, word_max_num), dtype='int32')
#         print data
        labels_list=[]
        i=0
        for instance in instances:
            labels_list.extend(self.get_array_of_type2(instance[0]))
            sent_num=len(instance)-1
            if sent_num>sent_max_num:
                sent_num=sent_max_num
            for j in range(1,sent_num+1):
                words=instance[j].split()
                word_num=len(words)
#                 print word_num
                if word_num> word_max_num:
                    word_num=word_max_num
                for k in range(word_num):
                    if self.word_2_index.has_key(words[k]):
                        data[i,j-1,k]=self.word_2_index[words[k]]
                    else:
                        data[i,j-1,k]=self.word_2_index['sparse_vectors']
            i+=1
                
        label_mat =  np.array(labels_list)
        label_mat = label_mat.reshape((len(instances), 2))
#         print label_mat
        return data,label_mat
    def represent_instances_3D_fea(self,instances,sent_max_num=None,word_max_num=None):
        data = np.zeros((len(instances), sent_max_num, word_max_num), dtype='int32')
        data_pos=np.zeros((len(instances), sent_max_num, word_max_num), dtype='int32')
        data_ner=np.zeros((len(instances), sent_max_num, word_max_num), dtype='int32')
#         print data
        labels_list=[]
        i=0
        for instance in instances:
            labels_list.extend(self.get_array_of_type2(instance[0]))
            sent_num=len(instance)-1
            if sent_num>sent_max_num:
                sent_num=sent_max_num
            for j in range(1,sent_num+1):
                segs=instance[j].split()
                word_num=len(segs)
#                 print word_num
                if word_num> word_max_num:
                    word_num=word_max_num
                for k in range(word_num):
                    segs_ele=segs[k].split('\\')
                    try:
                        words=segs_ele[0]
                        pos=segs_ele[1]
                        ner=segs_ele[2]
                    except:
                        print instance[j]
                        print segs[k]

                    if self.word_2_index.has_key(words):
                        data[i,j-1,k]=self.word_2_index[words]
                    else:
                        data[i,j-1,k]=self.word_2_index['sparse_vectors']
                    data_pos[i,j-1,k]=self.pos_fea_index[pos]
                    data_ner[i,j-1,k]=self.ner_fea_index[ner]

            i+=1
                
        label_mat =  np.array(labels_list)
        label_mat = label_mat.reshape((len(instances), 2))
#         print label_mat
        return data,data_pos,data_ner,label_mat

if __name__ == '__main__':
    pass
    
#    array_x, array_y = rep.generate_train_unit_rnn(test_x, test_y, 7)
#
#    array_3d = rep.generate_emb_based_data(array_x)


#    train = [line.strip() for line in open('C:/Users/IRISBEST/Desktop/CDR/data/test.txt')]
#    labels, words = rep.represent_instances_rnn(train, 'D')
#    for label, word in zip(labels, words):
#        print label
#        print word
#
#    train = [line.strip() for line in open('C:/Users/IRISBEST/Desktop/CDR/data/test.txt')]
    
#    labels_list = []
#    map = {'O_B':0,'O_I':0,'B_I':0,'B_O':0,'I_B':0,'I_O':0, 'I_I':0, 'B_B':0, 'O_O':0}
#    for elem in train:
#        id, labels, words = rep.represent_instance_D(elem)
#        for i in range(len(labels)-1):
#            if labels[i] == O and labels[i+1] == O:
#                map['O_O'] += 1
#            elif labels[i] == O and labels[i+1] == I:
#                map['O_I'] += 1
#            elif labels[i] == O and labels[i+1] == B:
#                map['O_B'] += 1
#            elif labels[i] == I and labels[i+1] == O:
#                map['I_O'] += 1
#            elif labels[i] == I and labels[i+1] == B:
#                map['I_B'] += 1
#            elif labels[i] == I and labels[i+1] == I:
#                map['I_I'] += 1
#            elif labels[i] == B and labels[i+1] == O:
#                map['B_O'] += 1
#            elif labels[i] == B and labels[i+1] == I:
#                map['B_I'] += 1
#            else:
#                map['B_B'] += 1
#    
#    print map
            
