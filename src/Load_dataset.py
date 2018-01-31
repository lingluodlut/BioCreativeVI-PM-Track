#encoding=utf-8
import numpy
def read_dataset(file):
    fin=open(file)
    token_list=[]
    ner_list=[]
    pos_list=[]
    for line in fin:
        line=line.strip()
        instance_token=[]
        instance_ner=[]
        instance_pos=[]
        instance_token.append(line)
        instance_ner.append(line)
        instance_pos.append(line)
        line=next(fin).strip()
        while(line!=''):
            segs=line.split('\t')
            instance_token.append(segs[0])
            instance_pos.append(segs[1])
            instance_ner.append(segs[2])
            line=next(fin).strip()
        token_list.append(' '.join(instance_token))
        ner_list.append(' '.join(instance_ner))
        pos_list.append(' '.join(instance_pos))
#    print ner_list
#    print token_list
    return token_list,pos_list,ner_list
def ensemble_read_dataset(infile):
    fin=open(infile)
    train_vec=[]
    lab_vec=[]
    for line in fin:
        segs=line.strip().split('\t')
        if segs[0]=='yes':
            lab_vec.append([1,0])
        else:
            lab_vec.append([0,1])
        vec_segs=segs[1].split()
        row=[float(vec_seg) for vec_seg in vec_segs]
        train_vec.append(row)
    x=numpy.array(train_vec)
    y=numpy.array(lab_vec)
    return x,y
if __name__=='__main__':
    test_token,test_pos,test_ner=read_dataset('/home/BIO/luoling/biocreativeVI/data/ppiac/fea-add/PMtask_Triage_DevSet_ir_scon.conll.PPIAC')
    print test_token[0],test_pos[0],test_ner[0]
