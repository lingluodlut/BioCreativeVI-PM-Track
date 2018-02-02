#encoding=utf-8
'''
Created on 2017年6月1日

@author: Administrator
'''
import xml.dom.minidom
from _codecs import encode
import random
import nltk
# parse trainingset from xml to text
def parse_xml_trainingset():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/test gold/PMtask_Triage_TestSet.xml'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/test gold/PMtask_Triage_TestSet_Gold.txt'
    fout=open(outfile,'w',encoding='utf-8')
    DOMTree=xml.dom.minidom.parse(infile)
    Data=DOMTree.documentElement
    documents=Data.getElementsByTagName('document')
    for document in documents:
        line=''
        pmid=document.getElementsByTagName('id')[0].childNodes[0].data
        relevant=document.getElementsByTagName('infon')[0].childNodes[0].data
        texts=document.getElementsByTagName('text')
        line=pmid+'\t'+relevant+'\t'
#         line=pmid+'\t'+'yes'+'\t'
#         t=0
        for text in texts:
#             t=t+1
            line=line+text.childNodes[0].data+'\t'
        fout.write(line.strip()+'\n')
#         if t<2:
#             print(relevant) 
#         print(abstract)
#         break
def trainset_div_testset():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/PMtask_Triage_TrainingSet.txt'
    trainfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_TrainingSet_ir.txt'
    devfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_DevSet_ir.txt'
    fin=open(infile,'r',encoding='utf-8')
    ftrain=open(trainfile,'w',encoding='utf-8')
    fdev=open(devfile,'w',encoding='utf-8')
    text_list=[]
    for line in fin:
        text_list.append(line)
    random.seed(1234)
    random.shuffle(text_list)
    i=0
    for line in text_list:
        i+=1
        if i<409:
            fdev.write(line)
        else:
            ftrain.write(line)
    fin.close()
    ftrain.close()
    fdev.close()
# def nltk_token():
#     infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_TrainingSet_ir.txt'
#     outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/token/PMtask_Triage_TrainingSet_ir.token'
#     fin=open(infile,'r',encoding='utf-8')
#     fout=open(outfile,'w',encoding='utf-8')
#     for line in fin:
#         line=line.strip()
#         segs=line.split('\t')
#         sentence=''
#         if len(segs)==4:
#             sentence=segs[2]+' '+segs[3]
#         else:
#             sentence=segs[2]
#         tokens = nltk.word_tokenize(sentence)
#         sentence_token=" ".join(tokens)    
#         fout.write(segs[0]+'\t'+segs[1]+'\t'+sentence_token+'\n')
#     fin.close()
#     fout.close()
def get_text():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/test/PMtask_Triage_TrainingSet.txt'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/test/PMtask_Triage_TrainingSet.text'
    fin=open(infile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    for line in fin:
        line=line.strip()
        segs=line.split('\t')
        sentence=''
        if len(segs)==4:
            sentence=segs[2]+' '+segs[3]
        else:
            sentence=segs[2]

        fout.write(sentence+'\n')
    fin.close()
    fout.close()
def produce_fastText_format():
    tokenfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/token/PMtask_Triage_TrainingSet_ir.token'
    labfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_TrainingSet_ir.txt'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/fasttext/PMtask_Triage_TrainingSet_ir.fastText'
    fin_token=open(tokenfile,'r',encoding='utf-8')
    fin_lab=open(labfile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    
    for line_token in fin_token:
        new_line=''
        line_lab=next(fin_lab)
        segs_lab=line_lab.strip().split('\t')
        if segs_lab[1]=='yes':
            new_line='__label__1 , '
        else:
            new_line='__label__0 , '
        new_line=new_line+line_token.lower()
        fout.write(new_line)
    fin_token.close()
    fin_lab.close()
    fout.close()
def produce_PPIACCNN_format():
    tokenfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/POS/PMtask_Triage_DevSet_ir_stop.pos_fea'
    labfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_DevSet_ir.txt'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/POS/PMtask_Triage_DevSet_ir_stop.pos_fea.PPIAC'
    fin_token=open(tokenfile,'r',encoding='utf-8')
    fin_lab=open(labfile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    
    for line_token in fin_token:
        new_line=''
        line_lab=next(fin_lab)
        segs_lab=line_lab.strip().split('\t')
        if segs_lab[1]=='yes':
            new_line='true '
#             new_line='1\t'
        else:
            new_line='false '
#             new_line='0\t'
        new_line=new_line+line_token.lower()
        fout.write(new_line)
    fin_token.close()
    fin_lab.close()
    fout.close()
#产生 HierarchicalRNN的数据格式，分词分句，第一行是标签，文档之间用空行间隔。
def produce_hierarchicalRNN_3Dformat():
    sstokenfile='H:/PHDwork/BioCreative VI/track4_PM/data/test/PMtask_TestSet.ssplit_token_scon'
    labfile='H:/PHDwork/BioCreative VI/track4_PM/data/test/PMtask_TestSet.txt'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/test/ppiac/PMtask_TestSet.hiernn_PPIAC'
    fin_token=open(sstokenfile,'r',encoding='utf-8')
    fin_lab=open(labfile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    max_sent_num=0.0
    ave_sent_num=0.0
    max_word_num=0.0
    ave_word_num=0.0
    i=0
    for line in fin_lab:
        i+=1
        segs=line.split('\t')
        if segs[1]=='no':
            fout.write('false\n')
        else:
            fout.write('true\n')
        text_line=next(fin_token).strip()
        sent_num=0
        while(text_line!=''):
            text_line=text_line.replace('(','-lrb-')
            text_line=text_line.replace(')','-rrb-')
            text_line=text_line.replace('[','-lsb-')
            text_line=text_line.replace(']','-rsb-')
            sent_num+=1
            words=text_line.split(' ')
            words_num=len(words)
            if words_num>max_word_num:
                max_word_num=words_num
            ave_word_num+=words_num
            fout.write(text_line+'\n')
            text_line=next(fin_token).strip()
        ave_sent_num+=sent_num
        if sent_num>max_sent_num:
            max_sent_num=sent_num
        fout.write('\n')
    print('max_sent:',max_sent_num,'ave_sent:',ave_sent_num/i,'max_word:',max_word_num,'ave_word',ave_word_num/ave_sent_num)
    fin_token.close()
    fin_lab.close()
    fout.close()
    
def svm_answer():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/BoW/PMtask_Triage_DevSet_ir.BoW'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/BoW/PMtask_Triage_DevSet_ir.answer'
    fin=open(infile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    for line in fin:
        segs=line.split('\t')
        if segs[0]=='1':
            fout.write('1\n')
        else:
            fout.write('-1\n')
    fin.close()
    fout.close()
#去掉停用词
def drop_stopword():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/PPIAC_CNN/PMtask_Triage_DevSet_ir.PPIAC'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/PPIAC_CNN/PMtask_Triage_DevSet_ir_stop_new.PPIAC'
    stopfile='H:/PHDwork/BioCreative VI/track4_PM/stopdict-new.txt'
    fin=open(infile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    fstop=open(stopfile,'r',encoding='utf-8')
    stop_dict=[]
    
    for line in fstop:
        line=line.strip()
        stop_dict.append(line)
    fstop.close()
    for line in fin:
        line=line.strip()
        segs=line.split(" ")
#         new_line=segs[0]
        new_line=''
        for i in range(len(segs)):
            if segs[i] not in stop_dict:
                new_line=new_line+" "+segs[i]
        fout.write(new_line.strip()+'\n')
    fout.close()
    fin.close()
#conll形式去掉停用词
def drop_stopword_conll():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/PMtask_Triage_TrainingSet_ir_scon.token_conll.PPIAC'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/PMtask_Triage_TrainingSet_ir_scon_stop.token_conll.PPIAC'
    stopfile='H:/PHDwork/BioCreative VI/track4_PM/stopdict-new.txt'
    fin=open(infile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    fstop=open(stopfile,'r',encoding='utf-8')
    stop_dict=[]
    
    for line in fstop:
        line=line.strip()
        stop_dict.append(line.lower())
    fstop.close()
    for line in fin:
        fout.write(line)
        line=next(fin).strip()
        while(line!=''):
            segs=line.split("\t")
            if segs[0] not in stop_dict:
                fout.write(line+'\n')
            line=next(fin).strip()
        fout.write('\n')
    fout.close()
    fin.close()
def fasttext_emb_addfirstline():
    fin=open('C:/Users/Administrator/Desktop/emb/BioCreativevi.vec','r',encoding='utf-8')
    fout=open('C:/Users/Administrator/Desktop/emb/BioCreativevi.vec_new','w',encoding='utf-8')
    num_line=next(fin).strip()
    segs=num_line.split()
    total=int(segs[0])+1
    print(total)
    fout.write(str(total)+" "+segs[1]+'\n')
    new_fistline='0_embedding '
    for i in range(int(segs[1])):
        new_fistline+="0 "
    fout.write(new_fistline+'\n')
    for line in fin:
        fout.write(line)
    fin.close()
    fout.close()

#直接从最后的输入文件中切'-'
def split_connector():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/test/PMtask_TestSet.ssplit_token'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/test/PMtask_TestSet.ssplit_token_scon'
    fin=open(infile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    new_line=''
    for line in fin:
        line=line.strip()
        line=line.replace('-lrb-','(')
        line=line.replace('-rrb-',')')
        line=line.replace('-lsb-','[')
        line=line.replace('-rsb-',']')
        line=line.replace('-',' - ')
        segs=line.split()
        new_line=" ".join(segs)
#         new_line=new_line.replace('(', '-lrb-')
#         new_line=new_line.replace(')','-rrb-')
        fout.write(new_line.strip()+'\n')
    fin.close()
    fout.close()
#产生PMID列表，进行pubtator的NER
def PMID_list():
    infile='H:/PHDwork/BioCreative VI/track4_PM/data/training/PMtask_Triage_TrainingSet.txt'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/PMtask_Triage_TrainingSet.pmid_list'
    fin=open(infile,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    for line in fin:
        line=line.strip()
        segs=line.split('\t')
        fout.write(segs[0]+'\n')
    fin.close()
    fout.close()
#从POS_conll文件直接产生最后的输入文件
def token_conll_inputfile():
    pos_conll_file='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/test/PMtask_Triage_TrainingSet.pos_token'
    train_label_file='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/test/PMtask_Triage_TrainingSet.txt'
    outfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/submit_until/test/PMtask_Triage_TrainingSet_all.token_conll.PPIAC'
    fin_pos_conll=open(pos_conll_file,'r',encoding='utf-8')
    fin_train_label=open(train_label_file,'r',encoding='utf-8')
    fout=open(outfile,'w',encoding='utf-8')
    for line in fin_train_label:
        line=line.strip()
        segs=line.split("\t")
        if segs[1]=='yes':
            fout.write('true\n')
        else:
            fout.write('false\n')
        pos_line=next(fin_pos_conll).strip()
        while(pos_line!=""):
            pos_line=pos_line.replace('(','-lrb-')
            pos_line=pos_line.replace(')','-rrb-')
            pos_line=pos_line.replace('[','-lsb-')
            pos_line=pos_line.replace(']','-rsb-')
            fout.write(pos_line.lower()+'\t'+'O\n')
            pos_line=next(fin_pos_conll).strip()
        fout.write('\n')
    fout.close()
    fin_pos_conll.close()
    fin_train_label.close()

def test_train_over():
    fin_train=open('H:/PHDwork/BioCreative VI/track4_PM/data/training/PMtask_Triage_TrainingSet.pmid_list','r',encoding='utf-8')
    fin_test=open('H:/PHDwork/BioCreative VI/track4_PM/data/test/PMtask_TestSet.pmid_list','r',encoding='utf-8')
    fout=open('H:/PHDwork/BioCreative VI/track4_PM/result/test/over_test_train','w',encoding='utf-8')
    dict_train=[]
    for line_train in fin_train:
        dict_train.append(line_train.strip())
    for line_test in fin_test:
        if line_test.strip() in dict_train:
            print(line_test.strip())
            fout.write(line_test)
        

               
if __name__ == "__main__":
    parse_xml_trainingset()
#     trainset_div_testset()
#     nltk_token()
#     get_text()
#     produce_fastText_format()
#     produce_PPIACCNN_format()
#     svm_answer()
#     drop_stopword()
#     drop_stopword_conll()   
#     fasttext_emb_addfirstline()
#     produce_hierarchicalRNN_3Dformat()
#     split_connector()
#     PMID_list()
#     token_conll_inputfile()
#     test_train_over()
