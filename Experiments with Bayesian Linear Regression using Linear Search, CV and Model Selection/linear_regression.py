# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:46:13 2019

@author: Sanket Pandilwar
"""
import sys
import numpy as np
import math
import matplotlib.pyplot as plt 
import time
import random


#Parsing design matrix phi
def parse_design_matrix(filename):
    return np.genfromtxt(filename, delimiter=',')
    
#Parsing labels
def parse_labels(filename):
    return np.genfromtxt(filename)

#def parse_test_set(filename):
#    phi_test = np.genfromtxt(filename)
#    return phi_test
#
#def parse_test_set_labels(filename):
#    t_test = np.genfromtxt(filename)
#    return t_test

#Function for calculating MSE
def calculate_MSE(phi,w,t):
    n=np.size(phi,0)
    mse=0
    for i in range(n):
        mse+=math.pow(np.dot(phi[i].T,w)-t[i],2)       
    mse=mse/n    
    return mse


# =============================================================================
#     Task 1: Regularization
# =============================================================================
def regularized(phi_train,t_train,phi_test,t_test):
    MSE_train=[]
    MSE_test=[]
    for lamb in range(0,151):
        phi_term=np.dot(phi_train.T,phi_train)
        adding_lamb=np.add(lamb*np.identity(phi_term.shape[0]),phi_term)
        inv_matrix=np.linalg.inv(adding_lamb)        
        temp2=np.dot(inv_matrix,phi_train.T)
        w=np.dot(temp2,t_train)       
        mse_train=calculate_MSE(phi_train,w,t_train)
        MSE_train.append(mse_train)
        mse_test=calculate_MSE(phi_test,w,t_test)
        MSE_test.append(mse_test)
    return MSE_train,MSE_test

# =============================================================================
# Task 2: Learning Curves
# =============================================================================
def learning_curve(phi_train,t_train,phi_test,t_test,lamb_list):
    n=np.size(phi_train,0)
    mse_for_each_lambda=dict()
    x_for_each_lambda=dict()
    for lamb in lamb_list:
        store_mean=[]
        store_x=[]
        for i in range(0,10):#for each train set size 
            x=[]
            mse=0.0
            for j in range(0,10):
                rand_index_list=random.sample(range(n),int(0.1*n*(i+1))) #creating random subsets
#                print(len(rand_index_list))
                training=[]
                t_training=[]
                for idx in rand_index_list:
                    training.append(phi_train[idx])
                    t_training.append(t_train[idx])
                phi_term=np.dot(np.array(training).T,np.array(training))
                adding_lamb=np.add(lamb*np.identity(phi_term.shape[0]),phi_term)
                inv_matrix=np.linalg.inv(adding_lamb)        
                temp2=np.dot(inv_matrix,np.array(training).T)
                w=np.dot(temp2,np.array(t_training))
                x.append(calculate_MSE(phi_test,w,t_test))
                mse+=calculate_MSE(phi_test,w,t_test)
            mse=mse/10
            store_x.append(x)
            store_mean.append(mse)
        mse_for_each_lambda[lamb]=store_mean
        if(lamb in x_for_each_lambda):
            x_for_each_lambda[lamb].append(store_x) 
        else:
            x_for_each_lambda[lamb]=store_x
            
    dict_for_std_each_lambda=dict()
    #calculating standard deviation
    for lamb in lamb_list:
        x_list=x_for_each_lambda.get(lamb)
        mean_list=mse_for_each_lambda.get(lamb)
        std_list=[]
        for i in range(0,10):
            std=0
            for j in range(0,10):
                std+=(x_list[i][j]-mean_list[i])**2
            std_list.append(math.sqrt(std/10))
        dict_for_std_each_lambda[lamb]=std_list
    
    
    return mse_for_each_lambda,dict_for_std_each_lambda


# =============================================================================
# Task 3.1: Model Selection using Cross Validation
# =============================================================================
def model_selection_using_cross_validation(phi_train,t_train,phi_test,t_test):
    start_time=time.time()
    k_fold=10 #since we are using 10 fold cross validation by default
    fold_size=int(len(phi_train)/k_fold)
    k_fold_dataset=[]
    k_fold_labels=[]
    #k-fold cross validation implementaion
    for k in range(k_fold):
        if (k!=(k_fold-1)):
            k_fold_dataset.insert(k,phi_train[k*fold_size:(fold_size)*(k+1)])
            k_fold_labels.insert(k,t_train[k*fold_size:(fold_size)*(k+1)])
        else: # if thers is extra data at the end and to avoid missing it, im putting that extra data in last fold
            k_fold_dataset.insert(k,phi_train[k*fold_size:])
            k_fold_labels.insert(k,t_train[k*fold_size:])
    
    
    dict_for_average_perfomance_of_each_lamb_on_folds=dict()
    #main implementaion of task 3.1
    for lamb in range(0,150):
        avg_performance=0.0
        for each_fold in range(k_fold):
            phi_test_data=k_fold_dataset[each_fold]
            t_test_data=k_fold_labels[each_fold]
            phi_train_data=[]     
            t_train_data=[]
            for j in range(k_fold):
                if j!=each_fold:
                    phi_train_data.extend(k_fold_dataset[j])
                    t_train_data.extend(k_fold_labels[j])      
            phi_term=np.dot(np.array(phi_train_data).T,np.array(phi_train_data))
            adding_lamb=np.add(lamb*np.identity(phi_term.shape[0]),phi_term)
            inv_matrix=np.linalg.inv(adding_lamb)        
            temp2=np.dot(inv_matrix,np.array(phi_train_data).T)
            w=np.dot(temp2,t_train_data)       
            avg_performance+=calculate_MSE(phi_test_data,w,t_test_data)
        dict_for_average_perfomance_of_each_lamb_on_folds[lamb]=avg_performance/k_fold
    
    min_lambda=min(dict_for_average_perfomance_of_each_lamb_on_folds, key=dict_for_average_perfomance_of_each_lamb_on_folds.get)
#    np.argmin(list(dict_name.values()))
#    print(dict_for_average_perfomance_of_each_lamb_on_folds)    
#temp=list(dict_for_average_perfomance_of_each_lamb_on_folds.values())
#    plt.plot(range(0,150),temp)
#    plt.show()
    min_mse=dict_for_average_perfomance_of_each_lamb_on_folds.get(min_lambda)
    print("\nMin MSE out of all lambda:",min_mse," and it's corresponding lambda value:",min_lambda)
   
    #calculating mse on test data from calculated lambda 
    phi_term=np.dot(phi_train.T,phi_train)
    adding_lamb=np.add(min_lambda*np.identity(phi_term.shape[0]),phi_term)
    inv_matrix=np.linalg.inv(adding_lamb)        
    temp2=np.dot(inv_matrix,phi_train.T)
    w=np.dot(temp2,t_train)
    mse_test=calculate_MSE(phi_test,w,t_test)
    print("MSE for part 3.1 on test data with lambda:",min_lambda," is",mse_test)
    total_time=time.time()-start_time
    print("Run time for part 3.1:",total_time," seconds\n")
    

# =============================================================================
# Task 3.2: Bayesian Model Selection
# =============================================================================
def bayesian_linear_regression(phi_train,t_train,phi_test,t_test):
    start_time=time.time()
    n=np.size(phi_train,0)
    gamma=0.0   
    alpha_random=np.random.randint(1,10)
    beta_random=np.random.randint(1,10)
    alpha_old=alpha_random
    beta_old=beta_random
    alpha_current=alpha_random
    beta_current=beta_random
    eigenvalues,eigenvector=np.linalg.eig(beta_current*np.dot(phi_train.T,phi_train))
    while(1):
#        eigenvalues,eigenvector=np.linalg.eig(beta_old*np.dot(phi_train.T,phi_train))
        alpha_old=alpha_current
        beta_old=beta_current
        phi_term=beta_old*np.dot(phi_train.T,phi_train)
        gamma=sum(eigenvalues/np.add(alpha_old,eigenvalues))
        sn_inv=np.add((alpha_old*np.identity(phi_term.shape[0])),phi_term)
        mn=np.dot(beta_old*np.linalg.inv(sn_inv),np.dot(phi_train.T,t_train))
        alpha_current=gamma/np.dot(mn.T,mn)
        beta_current=(n-gamma) *(1/sum(pow(np.dot(phi_train,mn.T)-t_train,2)))
        if((abs(alpha_current-alpha_old)<pow(10,-7)) and (abs(beta_current-beta_old)<pow(10,-7))):
            break
          
    lamb=alpha_current/beta_current
    print("Final Gamma:",gamma)
    print("Final Alpha:",alpha_current)
    print("Final Beta:",beta_current)
    print("Lambda for task 3.2:",lamb)
    
    #calculating mse on test data from calculated lambda 
    phi_term=np.dot(phi_train.T,phi_train)
    adding_lamb=np.add(lamb*np.identity(phi_term.shape[0]),phi_term)
    inv_matrix=np.linalg.inv(adding_lamb)        
    temp2=np.dot(inv_matrix,phi_train.T)
    mn=np.dot(temp2,t_train)
    mse_test=calculate_MSE(phi_test,mn,t_test)
    print("MSE for part 3.2 on test data:" , mse_test)
    total_time=time.time()-start_time
    print("Run time for part 3.2:",total_time," seconds\n")
    
if __name__ == "__main__":
    phi_train = parse_design_matrix(sys.argv[1])
    t_train= parse_labels(sys.argv[2])
    phi_test= parse_design_matrix(sys.argv[3])
    t_test= parse_labels(sys.argv[4])
    
    print("============== Task 1 ===================")
    mse_train,mse_test=regularized(phi_train,t_train,phi_test,t_test)
    print("min lambda:",np.argmin(mse_test)," \nmax lambda:",np.argmax(mse_test))

    lamb=[*range(0,151)]
    plt.plot(lamb, mse_train, linewidth=2.0,label="train",color='red')
    plt.plot(lamb,mse_test,linewidth=2.0,label="test",color='blue')
    plt.xlabel('Lambda') 
    plt.ylabel('Mean Square Error')
    plt.title('Train and Test MSE as a function of the regularization parameter lambda for file '+sys.argv[1]+' and '+sys.argv[3]) 
    plt.legend()
    plt.show()
    
    print("============== Task 2 ===================")
    lamb=[1,np.argmin(mse_test),np.argmax(mse_test)]
#    train_size=range(0,20)
    train_size=['0.1N','0.2N','0.3N','0.4N','0.5N','0.6N','0.7N','0.8N','0.9N','1N']
    mse_learning_curve_mean,std=learning_curve(phi_train,t_train,phi_test,t_test,lamb)
    plt.errorbar(train_size, mse_learning_curve_mean.get(lamb[0]),std.get(lamb[0]), linewidth=2.0,label="lambda=small-0",color='red')
    plt.errorbar(train_size, mse_learning_curve_mean.get(lamb[1]),std.get(lamb[1]), linewidth=2.0,label="lambda=just right-27",color='blue')
    plt.errorbar(train_size, mse_learning_curve_mean.get(lamb[2]),std.get(lamb[2]), linewidth=2.0,label="lambda=large-150",color='green')
    plt.xlabel('Train Set Size') 
    plt.ylabel('Mean Square Error')
    plt.title('MSE as a function of the size of the training set') 
    plt.legend()
    plt.show()
    
    print("============== Task 3.1 ===================")
    model_selection_using_cross_validation(phi_train,t_train,phi_test,t_test)
    
    print("============== Task 3.2 ===================")
    bayesian_linear_regression(phi_train,t_train,phi_test,t_test)
    print("Global Minima from part 1",mse_test[np.argmin(mse_test)])    