# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:04:11 2019

@author: Sanket Pandilwar
"""

'''

Number of topics K
Dirichlet parameter for topic distribution α, 
Dirichlet parameter for word distribution β, 
number of iterations to run sampler Niters, 
array of word indices w(n),
array of document indices d(n), 
array of initial topic indices z(n), 
where n = 1 . . . N words

'''

import numpy as np
import math
import matplotlib.pyplot as plt 
import time
import random
import copy

#data_folder="pp4data/artificial/"
data_folder="pp4data/20newsgroups/"
all_words_set=set()
all_unique_words=[]
all_duplicate_words=[]
w_n=[]
z_n=[]
d_n=[]
K=20
alpha=5/K
beta=0.01
n_iters=500
performance_global=[]
std_global=[]

def return_words_in_doc(filename):#reading each doc file
    with open(data_folder+filename, "r") as f:
        input_data=[]
        for line in f.read().split("\n"):
            for word in line.strip().split():
                input_data.append(word)
                all_duplicate_words.append(word)
                all_words_set.add(word)       
    return input_data

def parse_data(filename):#reading index.csv file which contains doc number and labels
    doc_word_dict=dict()
    label_dict=dict()
    data = np.genfromtxt(filename, delimiter=',',dtype=int)
    for i in range(len(data)):
        doc_word_dict[data[i][0]] = return_words_in_doc(str(data[i][0]))
        label_dict[data[i][0]] = data[i][1]
    return doc_word_dict,label_dict
   
    
def collapsed_gibbs_sampler(doc_word_dict,label_dict):
    
    num_of_docs=len(doc_word_dict)
    num_of_word = len(all_duplicate_words)
    cd=np.zeros((num_of_docs,K))
    ct=np.zeros((K,len(all_unique_words)))
    p=[0]*K
    
    
    pi_n=list(range(0,len(all_duplicate_words)))
    random.shuffle(pi_n)
    
    #initilizing word, topic and document indices
    for doc in doc_word_dict:
        for word in doc_word_dict[doc]:
            z_n.append(random.randint(0,K-1))
            d_n.append(doc-1)
            w_n.append(all_unique_words.index(word))
    
    #Initializing cd and ct
    for i in range(len(all_duplicate_words)):
        cd[d_n[i]][z_n[i]]+=1
        ct[z_n[i]][w_n[i]]+=1
        
    
    #Implementation of LDA using gibbs sampling        
    for i in range(0,n_iters):
        for n in range(0,num_of_word):
            word=w_n[pi_n[n]]
            topic=z_n[pi_n[n]]
            doc=d_n[pi_n[n]]
            cd[doc][topic]=cd[doc][topic]-1
            ct[topic][word]=ct[topic][word]-1
            
            for k in range(0,K): 
                p[k] =  (ct[k][word] + beta) * (cd[doc][k] + alpha) / ((len(all_unique_words) * beta + np.sum(ct[k,:])) * (K * alpha + np.sum(cd[doc,:]))) 

            p=np.divide(p,np.sum(p)) #normalizing
            topic=np.random.choice(range(0,K),p=p) #sampling
            z_n[pi_n[n]]=topic
            cd[doc][topic]=cd[doc][topic]+1
            ct[topic][word]=ct[topic][word]+1
    
    return z_n,cd,ct

def sigmoid(x): #calculates sigmoid
  return 1 / (1 + np.exp(-x))

#General implmentation to get first and second derivative for all GLM
def getderivatives(phi,d,alpha,w_old,R): 
    first_der_vector=np.dot(phi.T,d)-(alpha*w_old)
    sec_der_matrix=- (np.dot(phi.T,np.dot(R,phi)))-(alpha*(np.identity(phi.shape[1])))
    return first_der_vector,sec_der_matrix 

# Calculates and returns error rate, convergence time and number of iterations for logistic
def logistic(phi,t,phi_test,t_test,alpha):
    phi=np.asarray(phi)
    w=np.zeros((phi.shape[1],1))
    w_old=w
    iterations=0
    while(iterations!=100):
        yi=sigmoid(np.dot(phi,w_old))
        d= np.reshape(t,(len(t),1))-yi
        R=np.diagflat(yi * (1-yi))
        first_der_vector, sec_der_matrix = getderivatives(phi,d,alpha,w_old,R) 
        w_new=w_old-np.dot(np.linalg.inv(sec_der_matrix),first_der_vector)
        constraint=(np.linalg.norm(w_new-w_old,2))/np.linalg.norm(w_old,2) # Convergence condition of wmap
        if constraint<pow(10,-3):
            w_old=w_new
            break
        w_old=w_new
        iterations+=1
    
    w_map=w_old
    err_count=0
    t_hat = np.dot(np.array(phi_test),np.array(w_map))
    #Calculating error counts
    for i in range(t_hat.shape[0]):
        if(t_hat[i]>=0.5):
            t_hat[i]=1
        else:
            t_hat[i]=0
        if t_hat[i]!=t_test[i]:
            err_count+=1
            
    return err_count/t_hat.shape[0]

def evaluation(phi,t,alpha):# Values for approach (Logistic, Poisson, Ordinal)
    n=np.size(phi,0)
    list_for_errors=[]
    for i in range(0,30):
        rand_index_list=random.sample(range(n),n) #creating random subsets
        random_data=[]
        random_t=[]
        for idx in rand_index_list:
            random_data.append(phi[idx])
            random_t.append(t[idx])
        phi_test=random_data[:int(0.3*n)] # 30% of data to test set
        t_test=random_t[:int(0.3*n)]    
        phi_train=random_data[int(0.3*n):] # 70% of data to train set
        t_train=random_t[int(0.3*n):]
        size=len(phi_train)
        err_count_list=[]
        # Loop which stores average error rate, iterations and convergence time for respective approach and train set size
        for j in range(10):
            phi_sample=phi_train[0:int(0.1*size*(j+1))]
            t_sample=t_train[0:int(0.1*size*(j+1))]
            err_count=logistic(phi_sample,t_sample,phi_test,t_test,alpha)
            err_count_list.append(err_count)
        
        list_for_errors.insert(i,err_count_list)

    #Calculating average error rate for each train set size
    mean_for_each_train_set_size=[]
    for i in range(0,10):
        mn=0
        for j in range(30):
            mn+=list_for_errors[j][i]            
        mean_for_each_train_set_size.insert(i,mn/30)
    
    
    #Calculating standard deviation
    std_for_each_size=[]
    for i in range(10):
        std=0
        for j in range(30):
            std+=(list_for_errors[j][i]-mean_for_each_train_set_size[i])**2
        std_for_each_size.append(math.sqrt(std/30))
       
    performance=[]
    for error in mean_for_each_train_set_size:
        performance.append(1-error)
    
    performance_global.append(performance)
    std_global.append(std_for_each_size)
    
    
if __name__ == "__main__":
    start=time.time()
    doc_word_dict,label_dict = parse_data(data_folder+"index.csv")
    all_unique_words=list(all_words_set)
    z_n,cd,ct=collapsed_gibbs_sampler(doc_word_dict,label_dict)
    
    topic_word_dict={topic:dict() for topic in range(ct.shape[0])}
    
    topics={}
    for i in range(len(ct)):
        for j in ct[i].argsort()[-5:][::-1]:
            if i in topics:
                topics[i]+= [all_unique_words[j]]
            else:
                topics[i] = [all_unique_words[j]]
    
#    print("Topics:",topics)
    #writing topics and most frequent words for that topic into file
    with open("topicwords.csv","w") as f:
        for topic in topics:
            f.write(str(topics[topic]))
            f.write("\n")
    
    print("Time taken to run LDA:",time.time()-start, "seconds")
    
    #Calculating first representations
    first_representation=copy.deepcopy(cd)    
    for i in range(cd.shape[0]):
        for j in range(0,K):
            first_representation[i][j] = (cd[i][j] + alpha) / (K * alpha + np.sum(cd[i,:]))
      
    #Calculating second representations for bag of words    
    alpha=0.01        
    second_representation=np.zeros((cd.shape[0],ct.shape[1]),dtype=int)
    for doc in doc_word_dict:
        for word in doc_word_dict[doc]:
            second_representation[doc-1][all_unique_words.index(word)]+=1
    
    #labels
    labels=list(label_dict.values())
    labels=np.array(labels)
    np.reshape(labels,(len(labels),1))

    
    start=time.time()
    evaluation(first_representation,labels,alpha=0.01)
    print("Time taken to run Logistic for first representation:",time.time()-start, "seconds")
    
    start=time.time()
    evaluation(second_representation,labels,alpha=0.01)
    print("Time taken to run Logistic for first representation:",time.time()-start, "seconds")
    
    train_size=['0.1N','0.2N','0.3N','0.4N','0.5N','0.6N','0.7N','0.8N','0.9N','1N']
    plt.errorbar(train_size, performance_global[0],std_global[0], linewidth=2.0,label="LDA",color='orange')
    plt.errorbar(train_size, performance_global[1],std_global[1], linewidth=2.0,label="Bag of Words",color='blue')
    plt.xlabel('Train Set Size') 
    plt.ylabel('Performance')
    plt.title('LDA vs bag of words: Performance as a function of increasing training set')
    plt.legend()
    plt.show()  
        
        
        
