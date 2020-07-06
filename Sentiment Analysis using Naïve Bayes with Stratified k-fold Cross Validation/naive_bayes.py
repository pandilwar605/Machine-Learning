# -*- coding: utf-8 -*-
"""
@author: Sanket Pandilwar

"""

import sys
import random
import re
import matplotlib.pyplot as plt 
import math

def parse_file(filename):
    with open(filename, "r") as f:
        data=[]
        for line in f.read().split("\n"):
            if line.strip():
                data.append(line) 
        '''strip() in-built function of Python is used to remove all the leading and trailing spaces from a string.'''
        '''Taken code reference from websiyte: https://codereview.stackexchange.com/questions/145126/open-a-text-file-and-remove-any-blank-lines'''
        return data

def stratified_k_fold_dataset(k_fold,lines): 
# =============================================================================
#     this function implements stratified k fold and returns dataset 
#     which has k folds with each fold having equal ditributions of class data 
#     which helps to minimize a varinace/bias towards any one class
# =============================================================================

    class_0_doc=[]
    class_1_doc=[]
    dataset=[]
    document=[]
    dataset_0=[] #it will contain all the class 0 data and it's randomized
    dataset_1=[] #it will contain all the class 1 data and it's randomized
    for line in lines:
        document=line.split("\t")
        if int(document[1])==0:
            class_0_doc.append(line)
        else:
            class_1_doc.append(line)
    fold_size_for_0=int(len(class_0_doc)/k_fold)
    fold_size_for_1=int(len(class_1_doc)/k_fold)
    for k in range(k_fold):
        if (k!=(k_fold-1)):
            dataset_0.insert(k,class_0_doc[k*fold_size_for_0:(fold_size_for_0)*(k+1)])
            dataset_1.insert(k,class_1_doc[k*fold_size_for_1:(fold_size_for_1)*(k+1)])
        else: # if thers is extra data at the end and to avoid missing it, im putting that extra data in last fold
            dataset_0.insert(k,class_0_doc[k*fold_size_for_0:])
            dataset_1.insert(k,class_1_doc[k*fold_size_for_1:])

    for k in range(k_fold): #for each dfold,inserting data into dataset with equal distrubution  of classes
        dataset.insert(k,dataset_0[k]+dataset_1[k])
        
    for each_fold_data in dataset:
        random.shuffle(each_fold_data)

    return dataset

    
def learning_curve(k_fold,lines):
# =============================================================================
#     this function implements logic for dividing train data into 0.1N,0.2N...N 
#     and calculating accuracy and standard deviation for each of this train size with smoothing factors m=0 and m=1
# =============================================================================
    m=[0,1]
    random.shuffle(lines) #shuffling the data before proceeding
    dataset=stratified_k_fold_dataset(k_fold,lines)
    accuracy_for_each_fold_for_smoothing_0=[] #stores avg accuracy for each train set size for smoothing factor 0
    accuracy_for_each_fold_for_smoothing_1=[] #stores avg accuracy for each train set size for smoothing factor 1
    for smoothing_factor in m:
        for i in range(k_fold):
            accuracy_for_each_size=[]
            test_data=dataset[i]
            accuracy_for_each_size=[]
            train_data=[]      
            for j in range(k_fold):
                if j!=i:
                    train_data.extend(dataset[j])
            n=len(train_data)
            
            for j in range(0,10):
                training=train_data[0:int(0.1*n*(j+1))]
                document=[]
                total_documents=len(training)
                vocab=set()                 #it is a set collection to avoid entering duplicate words
                num_tokens_for_class_0=0
                num_tokens_for_class_1=0
                num_of_doc_0=0
                num_of_doc_1=0
                tokens_in_class_0=[]
                tokens_in_class_1=[]
                dict_for_class_0=dict()
                dict_for_class_1=dict()
                for doc in training: #all lines as list elements
                    document=doc.split("\t")   # split each row in comment and class
                    chars=" ".join(re.findall("[a-zA-Z]+", document[0])).lower()
                    for word in chars.split(" "):# split comment into number of words
                        vocab.add(word)
                        if int(document[1])==0:
                            tokens_in_class_0.append(word)
                            num_tokens_for_class_0 +=1
                        else:
                            tokens_in_class_1.append(word)
                            num_tokens_for_class_1 +=1
                    if int(document[1])==0:
                        num_of_doc_0 +=1
                    else:
                        num_of_doc_1 +=1
                        
                ML_for_0 = num_of_doc_0/(total_documents if total_documents!=0 else 1) # to avoid divide by zero error
                ML_for_1 = num_of_doc_1/(total_documents if total_documents!=0 else 1)
                # or we can use ML_for_1 = 1-ML_for_0
                
                for word in tokens_in_class_0: # to keep word and its count for each class 
                    dict_for_class_0[word]=dict_for_class_0.get(word,0)+1
                for word in tokens_in_class_1:
                    dict_for_class_1[word]=dict_for_class_1.get(word,0)+1
                    
                ''' This is for MAP of token parameters'''       
             
                MAP_of_token_parameters_0=dict()
                MAP_of_token_parameters_1=dict()
                for word in vocab: # This if else conditions are implemented to avoid divide by zero error
                    if((num_tokens_for_class_0+(smoothing_factor*len(vocab)))!=0):
                        MAP_of_token_parameters_0[word]= (dict_for_class_0.get(word,0)+smoothing_factor)/(num_tokens_for_class_0+(smoothing_factor*len(vocab)))
                    else:
                        MAP_of_token_parameters_0[word]=0
                    if((num_tokens_for_class_1+(smoothing_factor*len(vocab)))!=0):                        
                        MAP_of_token_parameters_1[word]= (dict_for_class_1.get(word,0)+smoothing_factor)/(num_tokens_for_class_1+(smoothing_factor*len(vocab)))
                    else:
                        MAP_of_token_parameters_1[word]=0
                
                count_of_right_predictions_per_fold=0
                total_predictions_made_per_fold=0
                doc=[]
                for doc in test_data: 
                    list_of_words_in_each_document=set()
                    predict_for_class_0=ML_for_0
                    predict_for_class_1=ML_for_1
                    test_document=doc.split("\t")   # split each row in comment and class
                    chars = " ".join(re.findall("[a-zA-Z]+", test_document[0])).lower()
                    for word in chars.split(" "):
                        list_of_words_in_each_document.add(word)
                    
                    for each_word in list_of_words_in_each_document:
                        predict_for_class_0*=(MAP_of_token_parameters_0.get(each_word,0)) if each_word in vocab else 1
                        predict_for_class_1*=(MAP_of_token_parameters_1.get(each_word,0)) if each_word in vocab else 1
                        '''
                        predict_for_class_0+=math.log((MAP_of_token_parameters_0.get(each_word,0)) if each_word in vocab else 1)
                        predict_for_class_1+=math.log((MAP_of_token_parameters_1.get(each_word,0)) if each_word in vocab else 1)
                        '''

                    actual_prediction=int(test_document[1])
                    observed_prediction=0 if predict_for_class_0>=predict_for_class_1 else 1
                    total_predictions_made_per_fold+=1
                    if (observed_prediction==actual_prediction):
                        count_of_right_predictions_per_fold+=1       
                accuracy=(count_of_right_predictions_per_fold/total_predictions_made_per_fold)*100

                accuracy_for_each_size.append(accuracy)
            if (smoothing_factor==0):
                accuracy_for_each_fold_for_smoothing_0.append(accuracy_for_each_size)
            else:
                accuracy_for_each_fold_for_smoothing_1.append(accuracy_for_each_size)
            
    return accuracy_for_each_fold_for_smoothing_0,accuracy_for_each_fold_for_smoothing_1

def experiment_1(accuracy_for_each_fold_for_smoothing_0,accuracy_for_each_fold_for_smoothing_1):
# =============================================================================
#     this function plots a graph for learning curve in experiment 1
# =============================================================================
    print(accuracy_for_each_fold_for_smoothing_0)
    abrakadabra_0=[] #combining accuracy results corresponding to train set size in one list for class 0 i.e. 0.1N -> [corresponding 10 results in one list], 0.2N-> [10 results in one list] 
    abrakadabra_1=[]
#    print(accuracy_for_each_fold_for_smoothing_0)
#    print(accuracy_for_each_fold_for_smoothing_1)
    average_per_size_0=[]
    average_per_size_1=[]
    std_per_size_0=[]
    std_per_size_1=[]
    for i in range(10):#seperating values of accuracy into corresponding train set size 
        temp_0=[]
        temp_1=[]
        for j in range(10):
            temp_0.append(accuracy_for_each_fold_for_smoothing_0[j][i])
            temp_1.append(accuracy_for_each_fold_for_smoothing_1[j][i])
        abrakadabra_0.insert(i,temp_0)
        abrakadabra_1.insert(i,temp_1)
#    print(abrakadabra_0)
#    print(abrakadabra_1)
        
    # calculating avergae accuracy and standard deviation for each train set size   
    for i in range(10):
        average_per_size_0.insert(i,sum(abrakadabra_0[i])/10)
        average_per_size_1.insert(i,sum(abrakadabra_1[i])/10)
        std_0=0
        std_1=0
        for j in range(10):
            std_0+=(average_per_size_0[i]-abrakadabra_0[i][j])**2
            std_1+=(average_per_size_1[i]-abrakadabra_1[i][j])**2
        std_per_size_0.insert(i,math.sqrt(std_0/10))
        std_per_size_1.insert(i,math.sqrt(std_1/10))
        
    train_size=['0.1N','0.2N','0.3N','0.4N','0.5N','0.6N','0.7N','0.8N','0.9N','1N']
    plt.errorbar(train_size,average_per_size_0, std_per_size_0,label="m=0")
    plt.errorbar(train_size,average_per_size_1, std_per_size_1,label="m=1")
    
    '''referred the plotting of errorbar from: https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html'''
    plt.xlabel('Train Set Size') 
    plt.ylabel('Average Accuracy per size in Percentage')
    plt.title('Accuracy and Std as a function of the train set size for file '+sys.argv[1]+' with '+str(k_fold)+' folds') 
    plt.legend()
    plt.show()
    
def Naive_Bayes_with_stratified_cross_validation(k_fold,dataset,smoothing_factor): 
# =============================================================================
#     this function implements stratified k-fold cross validation and also 
#     calculates all things needed for naive bayes like.. Ml,MAP,Avg Accuracy per fold,etc 
# =============================================================================
    global_list=[]
    for i in range(k_fold):
        test_data=[]
        train_data=[]
        test_data=dataset[i]
        for j in range(k_fold):
            if j!=i:
                train_data.extend(dataset[j])
        document=[]
        total_documents=len(train_data)
        vocab=set()                 #it is a set collection to avoid entering duplicate words
        num_tokens_for_class_0=0
        num_tokens_for_class_1=0
        num_of_doc_0=0
        num_of_doc_1=0
        tokens_in_class_0=[]
        tokens_in_class_1=[]
        dict_for_class_0=dict() #to store all the words and its count for further calculations
        dict_for_class_1=dict() #to store all the words and its count for further calculations
        for doc in train_data:  #iterating all rows in training data
            document=doc.split("\t")   # split each row in comment and class
            chars=" ".join(re.findall("[a-zA-Z]+", document[0])).lower() #remove all special characters
            #above line of code is refrred from website: https://stackoverflow.com/questions/8199398/extracting-only-characters-from-a-string-in-python
            for word in chars.split(" "):# split comment into number of words
                vocab.add(word)
                if int(document[1])==0:
                    tokens_in_class_0.append(word)
                    num_tokens_for_class_0 +=1
                else:
                    tokens_in_class_1.append(word)
                    num_tokens_for_class_1 +=1
            if int(document[1])==0:
                num_of_doc_0 +=1
            else:
                num_of_doc_1 +=1
                
        ML_for_0 = num_of_doc_0/(total_documents if total_documents!=0 else 1) # to avoid divide by zero error
        ML_for_1 = num_of_doc_1/(total_documents if total_documents!=0 else 1)
        
        for word in tokens_in_class_0: # to keep word and its count for each class 
            dict_for_class_0[word]=dict_for_class_0.get(word,0)+1
        for word in tokens_in_class_1:
            dict_for_class_1[word]=dict_for_class_1.get(word,0)+1
            
        ''' This is for MAP of token parameters'''       
     
        MAP_of_token_parameters_0=dict()
        MAP_of_token_parameters_1=dict()
        for word in vocab:      
            MAP_of_token_parameters_0[word]= (dict_for_class_0.get(word,0)+smoothing_factor)/(num_tokens_for_class_0+(smoothing_factor*len(vocab)))
            MAP_of_token_parameters_1[word]= (dict_for_class_1.get(word,0)+smoothing_factor)/(num_tokens_for_class_1+(smoothing_factor*len(vocab)))

        count_of_right_predictions_per_fold=0
        total_predictions_made_per_fold=0
        doc=[]
        for doc in test_data: # all rows as list elements
            list_of_words_in_each_document=set() #to avoid duplicate words, it is taken as set
            predict_for_class_0=ML_for_0 #initialize to ML, afterwards probabilities will be multiplied with this
            predict_for_class_1=ML_for_1
            
            test_document=doc.split("\t")   # split each row in comment and class
            chars = " ".join(re.findall("[a-zA-Z]+", test_document[0])).lower()
            for word in chars.split(" "):# split comment into number of words
                list_of_words_in_each_document.add(word)  #storing words without duplicates 
            
            for each_word in list_of_words_in_each_document: #predict a class for each document
                predict_for_class_0*=(MAP_of_token_parameters_0.get(each_word,0)) if each_word in vocab else 1
                predict_for_class_1*=(MAP_of_token_parameters_1.get(each_word,0)) if each_word in vocab else 1
        
            actual_prediction=int(test_document[1]) #test_document[1] contains actual class of document which i am comapring with observed class for the same document
            observed_prediction=0 if predict_for_class_0>predict_for_class_1 else 1
            total_predictions_made_per_fold+=1
            if (observed_prediction==actual_prediction):
                count_of_right_predictions_per_fold+=1       
        accuracy=(count_of_right_predictions_per_fold/total_predictions_made_per_fold)*100 #avergae percentage accuracy for the corresponding fold
        global_list.append(accuracy)
    return global_list
   
def experiment_2(k_fold,lines):
    '''Run stratied cross validation for Naive Bayes with smoothing parameter m = 0; 0:1; 0:2; : : : ; 0:9
    and 1; 2; 3; : : : ; 10 (i.e., 20 values overall). Plot the cross validation accuracy and standard
    deviations as a function of the smoothing parameter.'''
    m=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
    avg_accuracy=[]
    standard_deviation=[]
    for smoothing_factor in m:
        random.shuffle(lines) # randomly shuffling the data
        global_list=Naive_Bayes_with_stratified_cross_validation(k_fold,stratified_k_fold_dataset(k_fold,lines),smoothing_factor)
    
        total_accuracy=0.0
        for each_fold in range(len(global_list)):
            total_accuracy+=global_list[each_fold] 
        average_accuracy=(total_accuracy/len(global_list))
        std=[]
        for each_fold in range(len(global_list)):
            std.append((global_list[each_fold]-average_accuracy)**2)
        avg_accuracy.append(average_accuracy)
        standard_deviation.append(math.sqrt(sum(std)/len(global_list)))
 
        
#    print(avg_accuracy)
#    print(standard_deviation)
 
    plt.errorbar(m, avg_accuracy, standard_deviation, fmt='o') 
    
    plt.xlabel('Smoothing Factor') 
    plt.ylabel('Average Accuracy in Percentage')
    plt.title('Accuracy and Std as a function of the smoothing parameter for file '+sys.argv[1]+' with '+str(k_fold)+' folds') 
    plt.show()
    ''' referred the plotting of different graphs from follwing websites:
        https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
        https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
    '''


if __name__ == "__main__":
    data=parse_file(sys.argv[1]) #takes only filename as a input
    k_fold=10 #by default, all code is run on 10-fold, as mentioned in assignement
    
    accuracy_for_each_fold_for_smoothing_0,accuracy_for_each_fold_for_smoothing_1=learning_curve(k_fold,data) #this function returns the average accuracy for each train size(i.e 0.1N,0.2N...N with smoothing factors 0 and 1
    
    
    experiment_1(accuracy_for_each_fold_for_smoothing_0,accuracy_for_each_fold_for_smoothing_1)
    ''' =============================================================================
#     Experiment 1: For each of the 3 datasets run stratied cross validation to generate learning curves for Naive
#     Bayes with m = 0 and with m = 1. For each dataset, plot averages of the accuracy and
#     standard deviations (as error bars) as a function of train set size. It is insightful to put both
#     m = 0 and m = 1 together in the same plot.
    '''
    
    experiment_2(k_fold,data)
    
    ''' =============================================================================
#     Experiment 2:Run stratied cross validation for Naive Bayes with smoothing parameter m = 0; 0:1; 0:2; : : : ; 0:9
#     and 1; 2; 3; : : : ; 10 (i.e., 20 values overall). Plot the cross validation accuracy and standard
#     deviations as a function of the smoothing parameter.
# =============================================================================
    '''