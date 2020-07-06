# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:37:00 2019

@author: Sanket Pandilwar
"""
import sys
import numpy as np
import math
import matplotlib.pyplot as plt 
import time
import random


data_folder="pp3data/"

#Parsing design matrix phi
def parse_design_matrix(filename):
    return np.genfromtxt(data_folder+filename, delimiter=',')
    
#Parsing labels
def parse_labels(filename):
    return np.genfromtxt(data_folder+filename)

def sigmoid(x): #calculates sigmoid
  return 1 / (1 + np.exp(-x))

def sigmoid_for_ordinal(temp):
    if(np.isneginf(temp)):
        y = 0
    elif(np.isposinf(temp)):
        y = 1
    else:
        y = 1 / (1 + math.exp(-temp))
    return y

#General implmentation to get first and second derivative for all GLM
def getderivatives(phi,d,alpha,w_old,R): 
    first_der_vector=np.dot(phi.T,d)-(alpha*w_old)
    sec_der_matrix=- (np.dot(phi.T,np.dot(R,phi)))-(alpha*(np.identity(phi.shape[1])))
    return first_der_vector,sec_der_matrix 

# Calculates and returns error rate, convergence time and number of iterations for logistic
def logistic(phi,t,phi_test,t_test,alpha):
    phi=np.asarray(phi)
    w=np.zeros((phi.shape[1],1))
    start_convergence_time=time.time()
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
    
    convergence_time=time.time()-start_convergence_time
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
            
    return err_count/t_hat.shape[0],convergence_time,iterations


# Calculates and returns error rate, convergence time and number of iterations for poisson
def poisson(phi,t,phi_test,t_test,alpha):
    phi=np.asarray(phi)
    t=np.asarray(t)
    w=np.zeros((phi.shape[1],1))
    start_convergence_time=time.time()
    w_old=w
    iterations=0
    while(iterations<100):
        ai=np.dot(phi,w_old)
        yi=np.exp(ai)
        d= np.reshape(t,(len(t),1))-yi
        R=np.diagflat(yi)
        first_der_vector, sec_der_matrix = getderivatives(phi,d,alpha,w_old,R)
        w_new=w_old-np.dot(np.linalg.inv(sec_der_matrix),first_der_vector)
        if ((np.linalg.norm(w_new-w_old,2))/np.linalg.norm(w_old,2))<pow(10,-3): # Convergence condition of wmap
            break
        w_old=w_new
        iterations+=1
    
    convergence_time=time.time()-start_convergence_time
    w_map=w_old
    t_hat=np.floor(np.exp(np.dot(np.array(phi_test),np.array(w_map))))
    err = []
    for i in range(t_hat.shape[0]):
        err.append(abs(t_hat[i] - t_test[i]))
    return np.mean(err),convergence_time,iterations
   
    
# Calculates and returns error rate, convergence time and number of iterations for ordinal
def ordinal(phi,t,phi_test,t_test,alpha):    
    phi=np.asarray(phi)
    t=np.asarray(t)
    w=np.zeros((phi.shape[1],1))  
    s=1
    levels = [-np.Inf, -2, -1, 0, 1, np.Inf]   #no. of levels    
    start_convergence_time=time.time()
    w_old=w
    iterations=0
    while(iterations<100):
        ai=np.dot(np.asarray(phi),w_old)
        d = np.empty(len(t))
        R = np.empty(len(t))
        #loop for calculating R and d for ordinal using levels
        for i in range(len(t)):
            ti=int(t[i])
            temp1 = s* (levels[ti]- ai[i])
            y_i_ti = sigmoid_for_ordinal(temp1)
            temp2 = s* (levels[ti-1] - ai[i])
            y_i_ti_1 = sigmoid_for_ordinal(temp2)
            d[i] = y_i_ti + y_i_ti_1 - 1 
            R[i] = pow(s,2) *(y_i_ti * (1 - y_i_ti) + y_i_ti_1 * (1 - y_i_ti_1))
      
        d=d.reshape(len(d),1)
        R=np.diag(R)
        first_der_vector, sec_der_matrix = getderivatives(phi,d,alpha,w_old,R)
        w_new=w_old-np.dot(np.linalg.inv(sec_der_matrix),first_der_vector)
        if ((np.linalg.norm(w_new-w_old,2))/np.linalg.norm(w_old,2))<pow(10,-3): # Convergence condition of wmap
            break
        w_old=w_new
        iterations+=1
    
    convergence_time=time.time()-start_convergence_time
    w_map=w_old
    a=np.dot(np.array(phi_test),np.array(w_map))
    t_hat=[]
    for i in range(len(phi_test)):
        pj_list=[]
        for j in range(1,6):
            temp1=s* (levels[j] - a[i])
            yj=sigmoid_for_ordinal(temp1)
            
            temp2=s*(levels[j-1] - a[i])
            yj_1=sigmoid_for_ordinal(temp2)
            
            pj=yj - yj_1
            pj_list.append(pj)
        t_hat.append(pj_list.index(max(pj_list)) + 1)#since its index, we need to add 1 to get t_hat with right levels

    t_hat=np.asarray(t_hat)
    err = []
    #calculating absolute error and return average error rate
    for i in range(t_hat.shape[0]):
        err.append(abs(t_hat[i] - t_test[i]))
    
    return np.mean(err),convergence_time,iterations

#Plots the error rate as a function of training set sizes with error bars for the given approach and data 
#Also calculates average convergence time and average number of iterations for each training set size            
def evaluation(phi,t,alpha,approach):# Values for approach (Logistic, Poisson, Ordinal)
    n=np.size(phi,0)
    list_for_errors=[]
    list_for_time=[]
    list_for_iter=[]
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
        avg_time_list=[]
        avg_iter_list=[]
        # Loop which stores average error rate, iterations and convergence time for respective approach and train set size
        for j in range(10):
            phi_sample=phi_train[0:int(0.1*size*(j+1))]
            t_sample=t_train[0:int(0.1*size*(j+1))]
            if(approach=="Logistic"):
                err_count,avg_time,avg_iter=logistic(phi_sample,t_sample,phi_test,t_test,alpha)
            elif(approach=="Poisson"):
                err_count,avg_time,avg_iter=poisson(phi_sample,t_sample,phi_test,t_test,alpha)
            elif(approach=="Ordinal"):
                err_count,avg_time,avg_iter=ordinal(phi_sample,t_sample,phi_test,t_test,alpha)
            err_count_list.append(err_count)
            avg_time_list.append(avg_time)
            avg_iter_list.append(avg_iter)
        
        list_for_errors.insert(i,err_count_list) 
        list_for_time.insert(i,avg_time_list)
        list_for_iter.insert(i,avg_iter_list)

    #Calculating average error rate, iterations and convergence time for each train set size
    mean_for_each_train_set_size=[]
    avg_time_for_each_train_set_size=[]
    avg_iter_for_each_train_set_size=[]
    for i in range(0,10):
        mn=0
        tm=0
        it=0
        for j in range(30):
            mn+=list_for_errors[j][i]
            tm+=list_for_time[j][i]
            it+=list_for_iter[j][i]
            
        mean_for_each_train_set_size.insert(i,mn/30)
        avg_time_for_each_train_set_size.insert(i,tm/30)
        avg_iter_for_each_train_set_size.insert(i,it/30)
    
    #Calculating standard deviation
    std_for_each_size=[]
    for i in range(10):
        std=0
        for j in range(30):
            std+=(list_for_errors[j][i]-mean_for_each_train_set_size[i])**2
        std_for_each_size.append(math.sqrt(std/30))
    
    print("Average Convergence Time (in seconds):\n", avg_time_for_each_train_set_size)
    print("Average Number of Iterations:\n",avg_iter_for_each_train_set_size)
    
    train_size=['0.1N','0.2N','0.3N','0.4N','0.5N','0.6N','0.7N','0.8N','0.9N','1N']
    plt.errorbar(train_size, mean_for_each_train_set_size,std_for_each_size, linewidth=2.0,label=approach+" Regression",color='orange')
    plt.xlabel('Train Set Size') 
    plt.ylabel('Error Rate')
    plt.title('Error rate as a function of increasing training set')
    plt.legend()
    plt.show()
#    plt.savefig(approach)

def extra_credit(phi,t,approach):
    start_time=time.time()
    n=np.size(phi,0)
    list_for_errors=[]
    alpha_list=[]
    for i in np.arange(1,101): # Storing alpharanging from 1 to 100
        alpha_list.append(i)
        
    for i in range(0,30):
        rand_index_list=random.sample(range(n),n) #creating random subsets
        random_data=[]
        random_t=[]
        for idx in rand_index_list:
            random_data.append(phi[idx])
            random_t.append(t[idx])
        phi_test=random_data[:int(0.3*n)] # 30 % data for test
        t_test=random_t[:int(0.3*n)]    
        phi_train=random_data[int(0.3*n):] # 70% data for train
        t_train=random_t[int(0.3*n):]
        err_count_list=[]
        # Loop which stores average error rate, iterations and convergence time for respective approach and alpha values
        for alpha in alpha_list:
            if(approach=="Logistic"):
                err_count,avg_time,avg_iter=logistic(phi_train,t_train,phi_test,t_test,alpha)
            elif(approach=="Poisson"):
                err_count,avg_time,avg_iter=poisson(phi_train,t_train,phi_test,t_test,alpha)
            elif(approach=="Ordinal"):
                err_count,avg_time,avg_iter=ordinal(phi_train,t_train,phi_test,t_test,alpha)
            err_count_list.append(err_count)
        
        list_for_errors.insert(i,err_count_list)
    
    #calculating mean error rate for each alpha
    mean_for_each_alpha=[]
    for i in range(0,100):
        mn=0
        for j in range(30):
            mn+=list_for_errors[j][i]
            
        mean_for_each_alpha.insert(i,mn/30)
    
    plt.plot(alpha_list, mean_for_each_alpha, linewidth=2.0,label=approach+" Regression Extra Credit",color='red')
    plt.xlabel('Alphas') 
    plt.ylabel('Error Rate')
    plt.title('Error rate as a function of different alphas')
    plt.legend()
    plt.show()
    print("Alpha where error rate is minimum:", mean_for_each_alpha.index(min(mean_for_each_alpha))+1)
    print("Took",time.time()-start_time,"seconds")
    
if __name__ == "__main__":
    start_time=time.time()
    
    alpha=10
    data_matrix=parse_design_matrix(sys.argv[1])
    label=parse_labels(sys.argv[2])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)  #Adding new column of ones for w0 
    evaluation(new_data_matrix,label,alpha,"Logistic") # Calling logistic and plotting respective graph
    
    data_matrix=parse_design_matrix(sys.argv[3])
    label=parse_labels(sys.argv[4])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)
    evaluation(new_data_matrix,label,alpha,"Poisson") # Calling Poisson and plotting respective graph
    
    data_matrix=parse_design_matrix(sys.argv[5])
    label=parse_labels(sys.argv[6])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)
    evaluation(new_data_matrix,label,alpha,"Ordinal") # Calling Ordinal and plotting respective graph
    
    data_matrix=parse_design_matrix(sys.argv[7])
    label=parse_labels(sys.argv[8])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)
    evaluation(new_data_matrix,label,alpha,"Logistic") # Calling logistic for usps dataset and plotting respective graph
    
    
    ####### Extra Credit ###########################################################################################
    
    data_matrix=parse_design_matrix(sys.argv[1])
    label=parse_labels(sys.argv[2])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)  #Adding new column of ones for w0 
    extra_credit(new_data_matrix,label,"Logistic") # Calling logistic for getting alpha having minimum error rate and plotting graphs
    
    data_matrix=parse_design_matrix(sys.argv[3])
    label=parse_labels(sys.argv[4])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)
    extra_credit(new_data_matrix,label,"Poisson") # Calling logistic for getting alpha having minimum error rate and plotting graphs
    
    data_matrix=parse_design_matrix(sys.argv[5])
    label=parse_labels(sys.argv[6])
    new_data_matrix = np.insert(data_matrix,0,1,axis=1)
    extra_credit(new_data_matrix,label,"Ordinal") # Calling logistic for getting alpha having minimum error rate and plotting graphs

    ####### Extra Credit ###########################################################################################
    
#    print("Total time:",time.time()-start_time,"Seconds")