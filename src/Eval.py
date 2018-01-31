
import numpy as np



positive = 1
negtive = 0

def change_real2class(real_res_matrix):
    res_matrix = np.zeros_like(real_res_matrix, dtype=int)
    max_indexs = np.argmax(real_res_matrix, 1)
    for i in xrange(len(max_indexs)):
        res_matrix[i][max_indexs[i]] = 1
        
    return res_matrix

def change_real2class_vec(real_res_vec):
    res_vec = np.zeros_like(real_res_vec, dtype=int)
    max_index = np.argmax(real_res_vec)
    res_vec[max_index] = 1
    return res_vec

def change_5class_2_2class(five_class_matrix):
    res_matrix = np.zeros((five_class_matrix.shape[0],2), dtype=int)
    max_indexs = np.argmax(five_class_matrix, 1)
    for i in xrange(len(max_indexs)):
        if max_indexs[i]  <= 3:
            res_matrix[i][0] = 1
        else:
            res_matrix[i][1] = 1
        
    return res_matrix
    
 
def eval_mulclass4(ans_matrix, res_matrix,print_lag=False, real=True):
   
    #db:[214,298,94,278] ml:[7,62,2,24] both:[221,360,96,302]
    if ans_matrix.shape[1] == 5 or ans_matrix.shape[1] == 4:
        positives = [221,360,96,302]
    else:
        positives = [979]

    
    confuse_matrixs = np.zeros((ans_matrix.shape[1], 4))
    
    if real == True:
        res_matrix = change_real2class(res_matrix)
#        res_matrix = change_5class_2_2class(res_matrix)
    
#    FileUtil.writeFloatMatrix(res_matrix, './step2/both.res')
    
    class_indexs = np.argmax(ans_matrix, 1)
    for class_index in range(confuse_matrixs.shape[0]):
        for i in range(ans_matrix.shape[0]):
            if np.allclose(ans_matrix[i], np.zeros(4)):
                class_indexs[i] = -1
            if class_index == class_indexs[i]: #positive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][0] += 1 #TP
                else:
                    confuse_matrixs[class_index][1] += 1 #FN
            else: #negtive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][2] += 1 #FP
                else:
                    confuse_matrixs[class_index][3] += 1 #TN

    
    P, R = .0, .0    
    for i in range(confuse_matrixs.shape[0]):
        print confuse_matrixs[i]
        p = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][2])
#        r = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][1] + loss_positive[i])
        r = confuse_matrixs[i][0]/(positives[i])
        P += p
        R += r
        print 'Evaluation for the ' + str(i + 1) + 'th class'
        print 'P:    ', p
        print 'R:    ', r
        print 'F1:    ', 2*p*r/(p+r)
        print        
    P /= (confuse_matrixs.shape[0])
    R /= (confuse_matrixs.shape[0])
    F1 = 2*P*R/(P+R)
    print 'Evaluation for all the class'
    print 'P:    ', P
    print 'R:    ', R
    print 'F1:    ', F1
    print
    
    return P,R,F1

def eval_mulclass(ans_matrix, res_matrix,print_flag=False, real=True):
   
  
    confuse_matrixs = np.zeros((ans_matrix.shape[1], 4))
    
    if real == True:
        res_matrix = change_real2class(res_matrix)
    
    class_indexs = np.argmax(ans_matrix, 1)
    for class_index in range(confuse_matrixs.shape[0]):
        for i in range(ans_matrix.shape[0]):
            if class_index == class_indexs[i]: #positive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][0] += 1 #TP
                else:
                    confuse_matrixs[class_index][1] += 1 #FN
            else: #negtive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][2] += 1 #FP
                else:
                    confuse_matrixs[class_index][3] += 1 #TN

    
    P, R = .0, .0    
    for i in range(confuse_matrixs.shape[0]-1):
        if print_flag==True:
            print
	    print confuse_matrixs[i]
        p = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][2])
        r = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][1])
        P += p
        R += r
         
    P /= (confuse_matrixs.shape[0]-1)
    R /= (confuse_matrixs.shape[0]-1)
    F1 = 2*P*R/(P+R)
    Acc=(confuse_matrixs[0][0]+confuse_matrixs[0][3])/(confuse_matrixs[0][0]+confuse_matrixs[0][1]+confuse_matrixs[0][2]+confuse_matrixs[0][3])
    if print_flag==True:
        print 'Evaluation for all the class'
        print 'P=%.5f, R=%.5f, F=%.5f, ACC=%.5f' % (P,R,F1,Acc)
        print
    
    return [P,R,F1,Acc]





