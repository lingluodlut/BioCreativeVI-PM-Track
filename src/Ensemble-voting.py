#encoding=utf-8
'''
Created on 2017��8��4��

@author: Administrator
'''
def read_result_int(line):
    result=0
    line=line.strip()
    seg=line.split()
    if float(seg[0])>float(seg[1]):
        result=1
    else:
        result=-1
    return result

def read_result_float(line):
    result=0.0
    line=line.strip()
    seg=line.split()
    if float(seg[0])>float(seg[1]):
        result=float(seg[0])
    else:
        result=(-1)*float(seg[1])
    return result

def keras_eval(ensemble_result):
    PRF=[]
    answerfile='H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_DevSet_ir.txt'

    fin_answer=open(answerfile,'r',encoding='utf-8')
    
    TP = 0.0;#答案是正例结果是正例
    FP = 0.0;#答案是负例结果是正例
    FN = 0.0;#答案是正例结果是负例
    TN = 0.0;#答案是负例结果是负例
    id_num=0
    i=0
    for line in fin_answer:
        id_num+=1
#         print(id_num)
        answer=line.split('\t')[1]
        predict=ensemble_result[i]
        i+=1
        if predict==1:
            if answer=='yes':
                TP+=1
            
            else:
                FP+=1

                
        else:
            if answer=='yes':
                FN+=1

            else:
                TN+=1
  
        
    if(TP+FP==0):
            P=0.0
    else:
        P=TP/(TP+FP);
    R=TP/(TP+FN);
    if(R==0.0 or P==0.0):
        F=0.0
    else:
        F=2*P*R/(P+R);
    ACC=(TP+TN)/(TP+FP+FN+TN);
    print('P:',round(P,4),'R:',round(R,4),'F:',round(F,4),'ACC:',round(ACC,4))
    print('TP:',TP,' FN:',FN,' FP:',FP,' TN:',TN)
    fin_answer.close()
    PRF.append(round(P,4))
    PRF.append(round(R,4))
    PRF.append(round(F,4))
    PRF.append(round(ACC,4))
    return (PRF)

#加权投票，只使用0，1结果，不用概率值
def weight_vote_int(weight_dict):
    
#     result_path='H:/PHDwork/BioCreative VI/track4_PM/result/ensemble/'
#     fout_ensemble_result=open(result_path+'token_ensemble-0.6.result','w',encoding='utf-8')
#     fin1=open(result_path+'lstm-dev-scon-token-0.6.result','r',encoding='utf-8')
#     fin2=open(result_path+'rcnn-dev-scon-token-0.6.result','r',encoding='utf-8')
#     fin3=open(result_path+'bilstm-cnn-dev-scon-token-0.6.result','r',encoding='utf-8')
#     fin4=open(result_path+'cnn-dev-scon-token-0.6.result','r',encoding='utf-8')
#     fin5=open(result_path+'hiernn-dev-scon-token-new-0.6.result','r',encoding='utf-8')
#     fin6=open(result_path+'cnn-dev-scon-token-pos-ner.result','r',encoding='utf-8')
#     fin7=open(result_path+'bilstm-cnn-dev-scon-token-ner-pos.result','r',encoding='utf-8')


#     result_path='H:/PHDwork/BioCreative VI/track4_PM/result/test/'
#     fout_ensemble_result=open(result_path+'token_ensemble-0.6.result','w',encoding='utf-8')
#     fin1=open(result_path+'lstm-test-token-0.6.result','r',encoding='utf-8')
#     fin2=open(result_path+'rcnn-test-token-0.6.result','r',encoding='utf-8')
#     fin3=open(result_path+'bilstm-cnn-test-token-0.6.result','r',encoding='utf-8')
#     fin4=open(result_path+'cnn-test-token-0.6.result','r',encoding='utf-8')
#     fin5=open(result_path+'hiernn-test-token-0.6.result','r',encoding='utf-8')
#     fin6=open(result_path+'cnn-test-token-pos-ner.result','r',encoding='utf-8')
#     fin7=open(result_path+'bilstm-cnn-test-token-pos-ner.result','r',encoding='utf-8')

    result_path1='H:/PHDwork/BioCreative VI/track4_PM/result/post_challenge/dev/tune-fea/'
    result_path0='H:/PHDwork/BioCreative VI/track4_PM/result/post_challenge/dev/tune/'
    fout_ensemble_result=open(result_path1+'token_all_tune_ensemble_dev.result2','w',encoding='utf-8')
    fin1=open(result_path0+'lstm-token-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin2=open(result_path0+'cnn-token-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin3=open(result_path0+'lstm_cnn-token-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin4=open(result_path0+'rcnn-token-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin5=open(result_path0+'hiernn-token-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin6=open(result_path1+'lstm-token_pos_ner-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin7=open(result_path1+'cnn-token_pos_ner-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin8=open(result_path1+'lstm_cnn-token_pos_ner-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin9=open(result_path1+'rcnn-token_pos_ner-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')
    fin10=open(result_path1+'hiernn-token_pos_ner-ppipre-50d-tune.result_dev_post','r',encoding='utf-8')

#     result_path1='H:/PHDwork/BioCreative VI/track4_PM/result/post_challenge/test/tune-fea/'
#     result_path0='H:/PHDwork/BioCreative VI/track4_PM/result/post_challenge/test/tune/'
#     fout_ensemble_result=open(result_path1+'token_all_tune_ensemble_test.weight_vote_result','w',encoding='utf-8')
#     fin1=open(result_path0+'lstm-token-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin2=open(result_path0+'cnn-token-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin3=open(result_path0+'lstm_cnn-token-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin4=open(result_path0+'rcnn-token-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin5=open(result_path0+'hiernn-token-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin6=open(result_path1+'lstm-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin7=open(result_path1+'cnn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin8=open(result_path1+'lstm_cnn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin9=open(result_path1+'rcnn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
#     fin10=open(result_path1+'hiernn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')

#     weight_dict={
#                  'lstm':1,
#                  'cnn':1
#                  }
    ensemble_result=[]
    ele_result=0.0
    for line1 in fin1:
        
        if weight_dict['lstm']!=0:
            ele_result+=weight_dict['lstm']*read_result_int(line1)
            
        if weight_dict['cnn']!=0:
            line2=next(fin2)
            ele_result+=weight_dict['cnn']*read_result_int(line2)
        
        if weight_dict['lstm-cnn']!=0:
            line3=next(fin3)
            ele_result+=weight_dict['lstm-cnn']*read_result_int(line3)
        
        if weight_dict['rcnn']!=0:
            line4=next(fin4)
            ele_result+=weight_dict['rcnn']*read_result_int(line4)
        
        if weight_dict['hiernn']!=0:
            line5=next(fin5)
            ele_result+=weight_dict['hiernn']*read_result_int(line5)
            
        if weight_dict['lstm-fea']!=0:
            line6=next(fin6)
            ele_result+=weight_dict['lstm-fea']*read_result_int(line6)
             
        if weight_dict['cnn-fea']!=0:
            line7=next(fin7)
            ele_result+=weight_dict['cnn-fea']*read_result_int(line7)
        
        if weight_dict['lstm-cnn-fea']!=0:
            line8=next(fin8)
            ele_result+=weight_dict['lstm-cnn-fea']*read_result_int(line8)
        
        if weight_dict['rcnn-fea']!=0:
            line9=next(fin9)
            ele_result+=weight_dict['rcnn-fea']*read_result_int(line9)
            
        if weight_dict['hiernn-fea']!=0:
            line10=next(fin10)
            ele_result+=weight_dict['hiernn-fea']*read_result_int(line10)

        
        if ele_result>0:
            
            ele_result=0
            if weight_dict['lstm']!=0:
                ele_result+=weight_dict['lstm']*read_result_float(line1)
            
            if weight_dict['cnn']!=0:
                ele_result+=weight_dict['cnn']*read_result_float(line2)           
            if weight_dict['lstm-cnn']!=0:
                ele_result+=weight_dict['lstm-cnn']*read_result_float(line3)         
            if weight_dict['rcnn']!=0:
                ele_result+=weight_dict['rcnn']*read_result_float(line4)        
            if weight_dict['hiernn']!=0:
                ele_result+=weight_dict['hiernn']*read_result_float(line5)
            if weight_dict['lstm-fea']!=0:
                ele_result+=weight_dict['lstm-fea']*read_result_float(line6)            
            if weight_dict['cnn-fea']!=0:
                ele_result+=weight_dict['cnn-fea']*read_result_float(line7)
            if weight_dict['lstm-cnn-fea']!=0:
                ele_result+=weight_dict['lstm-cnn-fea']*read_result_float(line8)
            if weight_dict['rcnn-fea']!=0:
                ele_result+=weight_dict['rcnn-fea']*read_result_float(line9)
            if weight_dict['hiernn-fea']!=0:
                ele_result+=weight_dict['hiernn-fea']*read_result_float(line10)
            T_score=ele_result*0.5+0.5
            
            if ele_result<0:
#                 print("wrong1",ele_result)
                fout_ensemble_result.write("0.51111 0.48889"+'\n')
            else:
                fout_ensemble_result.write(str(round(T_score,5))+' '+str(round(1-T_score,5))+'\n')
            ele_result=1
            
        elif ele_result<0:
            
            ele_result=0
            if weight_dict['lstm']!=0:
                ele_result+=weight_dict['lstm']*read_result_float(line1)         
            if weight_dict['cnn']!=0:
                ele_result+=weight_dict['cnn']*read_result_float(line2)            
            if weight_dict['lstm-cnn']!=0:
                ele_result+=weight_dict['lstm-cnn']*read_result_float(line3)            
            if weight_dict['rcnn']!=0:
                ele_result+=weight_dict['rcnn']*read_result_float(line4)           
            if weight_dict['hiernn']!=0:
                ele_result+=weight_dict['hiernn']*read_result_float(line5)
            if weight_dict['lstm-fea']!=0:
                ele_result+=weight_dict['lstm-fea']*read_result_float(line6)            
            if weight_dict['cnn-fea']!=0:
                ele_result+=weight_dict['cnn-fea']*read_result_float(line7)
            if weight_dict['lstm-cnn-fea']!=0:
                ele_result+=weight_dict['lstm-cnn-fea']*read_result_float(line8)
            if weight_dict['rcnn-fea']!=0:
                ele_result+=weight_dict['rcnn-fea']*read_result_float(line9)
            if weight_dict['hiernn-fea']!=0:
                ele_result+=weight_dict['hiernn-fea']*read_result_float(line10)
            F_score=(-ele_result)*0.5+0.5
            
            if ele_result>0:
#                 print("wrong0",ele_result)    
                fout_ensemble_result.write("0.48889 0.51111"+'\n')  
            else:
                fout_ensemble_result.write(str(round(1-F_score,5))+' '+str(round(F_score,5))+'\n')          
            ele_result=0
        else:
            ele_result=0
            if weight_dict['lstm']!=0:
                ele_result+=weight_dict['lstm']*read_result_float(line1)         
            if weight_dict['cnn']!=0:
                ele_result+=weight_dict['cnn']*read_result_float(line2)            
            if weight_dict['lstm-cnn']!=0:
                ele_result+=weight_dict['lstm-cnn']*read_result_float(line3)            
            if weight_dict['rcnn']!=0:
                ele_result+=weight_dict['rcnn']*read_result_float(line4)           
            if weight_dict['hiernn']!=0:
                ele_result+=weight_dict['hiernn']*read_result_float(line5)
            if weight_dict['lstm-fea']!=0:
                ele_result+=weight_dict['lstm-fea']*read_result_float(line6)            
            if weight_dict['cnn-fea']!=0:
                ele_result+=weight_dict['cnn-fea']*read_result_float(line7)
            if weight_dict['lstm-cnn-fea']!=0:
                ele_result+=weight_dict['lstm-cnn-fea']*read_result_float(line8)
            if weight_dict['rcnn-fea']!=0:
                ele_result+=weight_dict['rcnn-fea']*read_result_float(line9)
            if weight_dict['hiernn-fea']!=0:
                ele_result+=weight_dict['hiernn-fea']*read_result_float(line10)
#             print('!!!0',ele_result)
            if ele_result>0.5:    
                ele_result=1
                fout_ensemble_result.write("0.51111 0.48889"+'\n')
            else:
                ele_result=0
                fout_ensemble_result.write("0.48889 0.51111"+'\n')
#             fout_ensemble_result.write("0"+'\n')
            
            
        ensemble_result.append(ele_result)
        ele_result=0.0
    PRF=keras_eval(ensemble_result)
    print(weight_dict)
    print()
    return(PRF)
#使用每一个类的概率值            
def weight_vote_float(weight_dict):
    result_path='H:/PHDwork/BioCreative VI/track4_PM/result/ensemble/'
    fin1=open(result_path+'lstm-dev-scon-token.result','r',encoding='utf-8')
    fin2=open(result_path+'rcnn-dev-scon-token.result','r',encoding='utf-8')
    fin3=open(result_path+'bilstm-cnn-dev-scon-token.result','r',encoding='utf-8')
    fin4=open(result_path+'cnn-dev-scon-token.result','r',encoding='utf-8')
    fin5=open(result_path+'hiernn-dev-scon-token-new.result','r',encoding='utf-8')
#     fin6=open(result_path+'cnn-dev-scon-token-pos-ner.result','r',encoding='utf-8')
#     fin7=open(result_path+'bilstm-cnn-dev-scon-token-ner-pos.result','r',encoding='utf-8')
    
#     weight_dict={
#                  'lstm':1,
#                  'cnn':1,
#                  'lstm-cnn':1
#                  }
    ensemble_result=[]
    ele_result=0.0
    for line1 in fin1:
        if weight_dict['lstm']!=0:
            ele_result+=weight_dict['lstm']*read_result_float(line1)
            
        if weight_dict['rcnn']!=0:
            line2=next(fin2)
            ele_result+=weight_dict['rcnn']*read_result_float(line2)
        
        if weight_dict['lstm-cnn']!=0:
            line3=next(fin3)
            ele_result+=weight_dict['lstm-cnn']*read_result_float(line3)
        
        if weight_dict['cnn']!=0:
            line4=next(fin4)
            ele_result+=weight_dict['cnn']*read_result_float(line4)
        
        if weight_dict['hie-lstm']!=0:
            line5=next(fin5)
            ele_result+=weight_dict['hie-lstm']*read_result_float(line5)
            
#         if weight_dict['cnn-pos-ner']!=0:
#             line6=next(fin6)
#             ele_result+=weight_dict['cnn-pos-ner']*read_result_float(line6)
#             
#         if weight_dict['lstm-pos-ner']!=0:
#             line7=next(fin7)
#             ele_result+=weight_dict['lstm-pos-ner']*read_result_float(line7)

        if ele_result>0:
            ele_result=1
        elif ele_result<0:
            ele_result=0
        else:
            print('!!!0')
            
        ensemble_result.append(ele_result)
        ele_result=0.0

    PRF=keras_eval(ensemble_result)
    print(weight_dict)
    print()
    return(PRF)            
#     print(ensemble_result)
def search_max():
    max_PRF=[]
    max_F=0
    i=0
    max_weight_dict={}
    Model_NUM=10
    if Model_NUM==10:
        while(i<=1):
            k=0
            while(k+i<=1):
                j=0
                while(k+i+j<=1):
                    f=0
                    while(k+i+j+f<=1):
                        g=0
                        while(k+i+j+f+g<=1):
                            h=0
                            while(k+i+j+f+g+h<=1):
                                e=0
                                while(k+i+j+f+g+h+e<=1):
                                    a=0
                                    while(k+i+j+f+g+h+e+a<=1):
                                        m=0
                                        while(k+i+j+f+g+h+e+a+m<=1):
                                            weight_dict={
                                             'lstm':round(i,3),
                                             'cnn':round(k,3),
                                             'lstm-cnn':round(j,3),
                                             'rcnn':round(f,3),
                                             'hiernn':round(g,3),
                                             'lstm-fea':round(h,3),
                                            'cnn-fea':round(e,2),
                                            'lstm-cnn-fea':round(a,3),
                                            'rcnn-fea':round(m,3),
                                            'hiernn-fea':round(1-k-i-j-f-g-h-e-a-m,3)}
                                            PRF=weight_vote_int(weight_dict)
                                            if PRF[2]>=max_F:
                                                max_F=PRF[2]
                                                max_PRF=PRF
                                                max_weight_dict=weight_dict
                                            m+=0.05
                                        a+=0.05
                                    e+=0.05
                                h+=0.05
                            g+=0.05
                        f+=0.05
                    j+=0.05
                k+=0.05
            i+=0.05
    else:
        while(i<=1):
            k=0.1
            while(k+i<=1):
                j=0.1
                while(k+i+j<=1):
                    f=0.1
                    while(k+i+j+f<=1):
                        weight_dict={
                         'lstm':0,#round(i,2),
                         'cnn':0,#round(k,2),
                         'lstm-cnn':0,#round(j,2),
                         'rcnn':0,#round(f,2),
                         'hiernn':0,#round(g,2),
                         'lstm-fea':round(i,2),
                        'cnn-fea':round(k,2),
                        'lstm-cnn-fea':round(j,2),
                        'rcnn-fea':round(f,2),
                        'hiernn-fea':round(1-i-k-j-f,2)}
                        PRF=weight_vote_int(weight_dict)
                        if PRF[2]>=max_F:
                            max_F=PRF[2]
                            max_PRF=PRF
                            max_weight_dict=weight_dict
                        f+=0.1
                    j+=0.1
                k+=0.1
            i+=0.1
    print("ensemble result:")
    print(max_PRF)
    print(max_weight_dict)
    
def result_ensemble():
    tag='test'
    path='H:/PHDwork/BioCreative VI/track4_PM/result/post_challenge/test/tune-fea/'
#     fin_lab=open('H:/PHDwork/BioCreative VI/track4_PM/data/training/train_and_dev/PMtask_Triage_DevSet_ir.txt','r',encoding='utf-8')
    fin_lab=open('H:/PHDwork/BioCreative VI/track4_PM/data/test gold/PMtask_TestSet_gold.hiernn_PPIAC','r',encoding='utf-8')
    fin_cnn=open(path+'cnn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
    fin_lstm=open(path+'lstm-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
    fin_rcnn=open(path+'rcnn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
    fin_lstm_cnn=open(path+'lstm_cnn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
    fin_hie=open(path+'hiernn-token_pos_ner-ppipre-50d-tune.result_test_post','r',encoding='utf-8')
    fout=open(path+'ensemble_nn_pos_ner-50d.test','w',encoding='utf-8')
    for line_lstm in fin_lstm:
        if tag=='test':
            line_lab=next(fin_lab).strip()
            if line_lab=='true':
                lab='yes'
            else:
                lab='no'
            while(line_lab!=''):
                line_lab=next(fin_lab).strip()
        else:
            line_lab=next(fin_lab).strip()
            seg=line_lab.split('\t')
            lab=seg[1]
        line_rcnn=next(fin_rcnn)
        line_lstm_cnn=next(fin_lstm_cnn)
        line_cnn=next(fin_cnn)
        line_hie=next(fin_hie)
        fout.write(lab+'\t'+line_lstm.strip()+' '+line_cnn.strip()+' '+line_lstm_cnn.strip()+' '+line_rcnn.strip()+' '+line_hie.strip()+'\n')
def result_ensemble_all():

    path='H:/PHDwork/BioCreative VI/track4_PM/result/post_challenge/all/'

    fin=open(path+'ensemble_nn-50d.dev','r',encoding='utf-8')
    fin_fea=open(path+'ensemble_nn_pos_ner-50d.dev','r',encoding='utf-8')
    fout=open(path+'ensemble_nn_all-50d.dev','w',encoding='utf-8')
    for line in fin:
        line=line.strip()
        seg=line.split('\t')
        line_fea=next(fin_fea).strip()
        seg_fea=line_fea.split('\t')

        fout.write(seg[0]+'\t'+seg[1].strip()+' '+seg_fea[1].strip()+'\n')

def weight_sum():
    weight_dict={
                 'lstm':0.1,
                 'cnn':0.2,
                 'lstm-cnn':0,
                 'rcnn':0.4,
                 'hiernn':0.2,
                 'lstm-fea':0,
                 'cnn-fea':0,
                 'lstm-cnn-fea':0,
                 'rcnn-fea':0,
                 'hiernn-fea':0.1
                 }
#     weight_dict={
#                  'lstm':0.3,
#                  'rcnn':0,
#                  'lstm-cnn':0.3,
#                  'cnn':0,
#                  'hie-lstm':0.2,
#                  'cnn-pos-ner':0.1,
#                  'bilstm-pos-ner':0.1
#                  }
    PRF=weight_vote_int(weight_dict)
    print(PRF)
        
if __name__ == "__main__":
#     weight_vote_int()
#     weight_vote_float()
    search_max()
#     result_ensemble()
#     weight_sum()
#     result_ensemble_all()