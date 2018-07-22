# Implementation for 
# Comparator-learning Combinatorial Functions (JMLR, 2016)
# @author: Yuhao Wan

import numpy as np
import pandas as pd
from functools import cmp_to_key
import math
import copy
from sklearn.linear_model import perceptron
from sklearn.metrics import accuracy_score

class pairwise_oracle():
  # Implementation of pairwise comparison oracle 
  
    def __init__(self, preference_list):    
        """ taken in an ordered list as given preference
        the list is in ascending order (so later in the list means more preferred)
        assuming strict preference for now """
        self.preference = np.array(preference_list)
        
    def compare(self, s1, s2):
        """ output -1 or 1 according to the preference list
        # s1 > s2 --> 1; s1 < s2 --> -1 """
        i, j = np.where(self.preference == s1), np.where(self.preference == s2)
        return 1 if i > j else -1

    def batch_compare(self, s_table):
        """ Input dim: k x 2 matrix, each row is (s1, s2)
        Return a vector of length k according to the preference list """
        vectorCompare = np.vectorize(self.compare)
        return vectorCompare(s_table.T[0], s_table.T[1])

class comparison_learner():
  # Implementation of a comparison learning function
  
    def __init__(self, S, n, oracle, epsilon, delta): 
        """initialize with 
        # a training sample S, ground set size n
        # an oracle
        # initial dictionary R[(i, j)] = [w, theta]
        # hyper-parameters epsilon, delta""" 

        self.S = S
        self.n = n
        self.oracle = oracle
        self.R = {}
        self.epsilon = epsilon
        self.delta = delta
        self.m = self.calculate_m(epsilon, delta)
        
        
    def train(self):
        """main function, updates dictionary R with weights""" 
        
        # step 1
        S1 = np.random.choice(self.S, self.m, replace = False) 
        S2 = np.setdiff1d(self.S, S1) 
        
        # step 2
        S1_sorted = self.sort_1d(S1) 
        
        # step 3
        S2_dict = self.sort_2d(S1_sorted, S2)
        
        # step 4 & 5
        R = self.get_weights(S1_sorted, S2_dict)
        
        # step 6
        self.R = self.minimalize(R)
                
            
    def predict(self, s1, s2):
        """Input s1, s2. Returns comparison result. (s1 > s2 --> 1; s1 < s2 --> -1)"""
        
        chi_s1 = np.append(num_to_ind(s1, self.n), 1) # add intercept term
        chi_s2 = np.append(num_to_ind(s2, self.n), 1)

        for (a, b) in self.R:
            weight = self.R[(a, b)]
            if np.dot(weight, chi_s1) < 0 and np.dot(weight, chi_s2) > 0:
                return -1
        return 1
    
    def batch_predict(self, test_sample):
        """Input dimension: K x 2. Returns a vector of length K comparison result"""
        return [self.predict(row[0], row[1]) for row in test_sample]
        
    def test_accuracy(self, test_sample, test_truth):
        """Input vector of length k. Returns accuracy rate"""
        predict_results = self.batch_predict(test_sample)
        accuracy = sum(np.equal(predict_results, test_truth)) / len(test_truth)
        return accuracy
    
    def calculate_m(self, epsilon, delta):
        """Input hyperparameters. Returns m"""
        return int((2/epsilon)*math.log(1/(epsilon*delta)))
    
    def sort_1d(self, S1):
        """Sorts S1 by value function"""
        return sorted(S1, key=cmp_to_key(self.oracle.compare))
    
    
    def sort_2d(self, S1, S2):
        """Sorts S2 2-dimensionally"""
        S2_dict = {}
        for i in range(self.m):
            for j in range(i+1, self.m):
                S2_dict[(i, j)] = []
                for s in S2:
                    # add into S2_dict if s < s_i or s > s_j
                    if self.oracle.compare(s, S1[i]) == -1 or self.oracle.compare(s, S1[j]) == 1:
                        S2_dict[(i, j)].append(s)
        return S2_dict
    
    
    def minimalize(self, R):
        """Returns a minimalized dictionary R to contain only minimal pairs""" 
        R_copy = copy.deepcopy(R)
        for (a, b) in R_copy:
            for (c, d) in R_copy:
                if (a, b)!= (c, d) and a <= c and b >= d and (a, b) in R:    
                    del R[(a, b)]
        return R
    
    
    def get_weights(self, S1, S2_dict):
        """Returns a dictionary R where key is (i, j) and value is the weights [w_ij, theta_ij]"""
        R = {}
        for i in range(self.m):
            for j in range(i+1, self.m):
                S_ij = S2_dict[(i, j)]
                labels = [-1 if self.oracle.compare(s, S1[i]) == -1 else 1 for s in S_ij]
                S_ij = set_to_matrix(S_ij, self.n)
                net = perceptron.Perceptron(max_iter=100, 
                                            fit_intercept=True, 
                                            eta0=0.002)
                if len(np.unique(labels)) == 2: # fit if having two unique labels  
                    net.fit(S_ij, labels)
                    if net.score(S_ij, labels) == 1: # add to R if no training error
                        w = net.coef_[0]
                        theta = net.intercept_[0]
                        R[(i, j)] = np.append(w, theta)
        return R

# Helper functions    
def num_to_ind(k, n): 
    """Input number. Returns a list of binary with length n."""
    length = '0' + str(n) + 'b'
    return np.array([int(x) for x in format(k, length)])

def ind_to_num(ind_vec):
    """Input a list of binary with length n. Returns a number."""
    return np.array(int(''.join(map(str, list(ind_vec))), 2))

def set_to_matrix(num_list, n):
    """Input set of numbers. Returns a matrix where each row is a indicator vector."""
    return np.array([num_to_ind(x, n) for x in num_list])

def run_test(n, epsilon, delta, S_size, pref_list, test_size):
    """Input learning model hyper-parameters and test parameters. Return accuracy rate."""
    # build pair-comparison oracle
    oracle = pairwise_oracle(pref_list)

    # build comparison learner
    S = np.random.choice(pref_list, S_size, replace = False)
    myLearner = comparison_learner(S, n, oracle, epsilon, delta)
    
    # train
    myLearner.train()
    
    # test
    select_sample = np.random.choice(pref_list, 2*test_size, replace = False)
    test_sample = [select_sample[0:test_size], select_sample[test_size:2*test_size]]
    test_sample = np.rot90(test_sample) # k x 2 dimension
    test_truth  = oracle.batch_compare(test_sample)
    accuracy    = myLearner.test_accuracy(test_sample, test_truth)

    return accuracy

if __name__ == "__main__":
    run_test(n=10, epsilon=0.2, delta=0.2, S_size=200, 
        pref_list=range(1024), test_size=100)
