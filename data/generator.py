from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import random
from torch.autograd import Variable
from . import sensor
import os
from . import parser
import pickle
import pandas as pd
from collections import Counter

class Generator(data.Dataset):
    def __init__(self, root, args, partition='train', dataset='sensor'):
        self.root = root
        self.partition = partition  
        self.args = args
        self.partition = partition

        self.dataset = dataset
        if self.dataset == 'sensor':
            self.input_channels = 3
            self.size = (224, 224)

        if dataset == 'sensor':
            self.loader = sensor.Sensor(self.root)
            self.data, self.label_encoder,self.label_decoder = self.loader.load_dataset(self.partition, args, self.size) 
            self.emp_distribution,self.emp_prob = self.calc_distribution() 
            self.class_size = len(self.data)

        else:
            raise NotImplementedError

        self.class_encoder = {}
        for id_key, key in enumerate(self.data):
            self.class_encoder[key] = id_key    
            
    def calc_distribution(self):  
    
        """Function used to calculate sampling probabilities """   
        distribution = {}
        
        for i in self.data:
            distribution[i] = len(self.data[i])
            
        counts = Counter(distribution)
        probabilities = []
        counter = 0
        
        for s in counts.values():
            if counter < len(counts)-1:
                probabilities.append(s/sum(distribution.values()))
            else:
                probabilities.append(1-sum(probabilities))
            counter = counter +1
        return distribution,probabilities


    def get_task_batch(self, batch_size=15, cuda=False, variable=False): 

        """Function used to perform sampling
        
        Parameters:
        batch_size (int): batch size used
        cuda: Whether or nut CUDA can be used
        variable: If CUDA cannot be used 
        
        Returns:
        return_arr (List): contains all data created such as sampled batches, unlabeled sample and labels
        labels_dict (dict) : contains true labels needed for testing
        """   
        
        #Used to smaple batches from dataset                                                                                      
        support_size = 15      
        n_way = len(self.data) 
        
        batch_x = np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32')
        labels_x = np.zeros((batch_size, n_way), dtype='float32')
        labels_x_global = np.zeros(batch_size, dtype='int64')
             
        labels_dict = np.zeros((batch_size, n_way), dtype='float32')
        batches_xi, labels_yi = [], []

        for i in range(support_size):
            batches_xi.append(np.zeros((batch_size, self.input_channels, self.size[0], self.size[1]), dtype='float32'))
            labels_yi.append(np.zeros((batch_size, n_way), dtype='float32'))
        # Iterate over tasks for the same batch
        for batch_counter in range(batch_size):       
            c = Counter()
            sampled_list = []
            for _ in range(support_size):                                         
                profession = np.random.choice(list(self.emp_distribution.keys()), p=self.emp_prob)
                c[profession] += 1
                sampled_list.append(profession)

            classes_ = list(self.data.keys())                         
            classes_ = random.sample(classes_, n_way) 
   
            labels_dict[batch_counter,:] =classes_   
            positive_class = random.sample(sampled_list,1)[0]
            positive_class_index = classes_.index(positive_class)
            indexes_perm = np.random.permutation(support_size)
            counter = 0

            for class_counter, class_ in enumerate(classes_):     

                if class_counter == positive_class_index:
                    
                    samples = random.sample(self.data[class_], c[class_]+1)
                    batch_x[batch_counter, :, :, :] = samples[0]
                    labels_x[batch_counter, class_counter] = 1
                    labels_x_global[batch_counter] = self.class_encoder[class_]
                    samples = samples[1::]
                else:

                    samples = random.sample(self.data[class_], c[class_])

                for s_i in range(0, len(samples)):

                    batches_xi[indexes_perm[counter]][batch_counter, :, :, :] = samples[s_i]
                    labels_yi[indexes_perm[counter]][batch_counter, class_counter] = 1
  
                    counter += 1

        batches_xi = [torch.from_numpy(batch_xi) for batch_xi in batches_xi]
        labels_yi = [torch.from_numpy(label_yi) for label_yi in labels_yi]
        labels_x_scalar = np.argmax(labels_x, 1)
        return_arr = [torch.from_numpy(batch_x), torch.from_numpy(labels_x), torch.from_numpy(labels_x_scalar),
                      torch.from_numpy(labels_x_global), batches_xi, labels_yi]
        if cuda:
            return_arr = self.cast_cuda(return_arr)
        if variable:
            return_arr = self.cast_variable(return_arr)
        return return_arr,labels_dict

    def cast_cuda(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_cuda(input[i])
        else:
            return input.cuda()
        return input

    def cast_variable(self, input):
        if type(input) == type([]):
            for i in range(len(input)):
                input[i] = self.cast_variable(input[i])
        else:
            return Variable(input)

        return input
