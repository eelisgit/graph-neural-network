#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import pickle

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor

def gmul(input):

    """Function used to perform matrix-matrix multiplication on graph and edges.
        
    Parameters:
    input (list): List which contains edges and nodes of graph.
            
    Returns:
    Output (Tensor): Tensor of diemsnions [Batch_size x (Images +1) x Expansion size].
    
    """    
    
    W, x = input
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) 
    output = torch.bmm(W, x) 
    output = output.split(N, 1)
    output = torch.cat(output, 2) 

    return output

#Graph Convolution
class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
    
        """Function used to calculate distance between nodes in graph.
        
        Parameters:
            input (list): List which contains edges and nodes of graph.
            
        Returns:
            W (Tensor): Tensor of dimensions [Batch_size x (Images +1) x (Images +1) x 2]
            x (Tensor): Tensor of dimensions [Batch_size x (Images +1) x 48] where 48 is the growth rate of network.

        """    
        
        W = input[0]
        
        #Tensor multiplication of edges and nodes
        x = gmul(input) 
      
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)

        x = self.fc(x) 
        
        if self.bn_bool:
            x = self.bn(x)
        
        x = x.view(*x_size[:-1], self.num_outputs)
        

        return W, x


class Wcompute(nn.Module):

    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):    
    
        """Function used to calculate distance between nodes in graph.
        
        Parameters:
            x (tensor): Tensor of diemsnions [Batch_size x (Images +1) x Features] 
            W_id (tensor): Tensor of diemsnions [Batch_size x (Images +1) x (Images +1) x 1] 
        Returns:
            W_new (Tensor): Tensor of diemsnions [Batch_size x (Images +1) x (Images +1) x 2]

        """    

        W1 = x.unsqueeze(2)                                                           
        W2 = torch.transpose(W1, 1, 2)      
        
        #Use absolute value as distance metrix
        W_new = torch.abs(W1 - W2) 
        W_new = torch.transpose(W_new, 1, 3) 
        
        #Pass distances into neural network
        W_new = self.conv2d_1(W_new)
       
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) 
        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)               

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)   

        else:
            raise(NotImplementedError)
        
        return W_new


class GNN_nl(nn.Module):                                           
    def __init__(self, args, input_features, nf, J,final_output):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        #Two layers in GNN + final layer
        self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, final_output, 2, bn_bool=False)

    def forward(self, x):     

        """Function used to calculate distance and then pass nodes and edges into graph convolution.
        
        Parameters:
            x (tensor): Tensor of diemsnions [Batch_size x (Images +1) x Features] 

        Returns:
            out (Tensor): Tensor of shape [Batch_size x Classes]  giving logits for unlabeled sample.

        """   
        
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))  
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)     

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])   

            x = torch.cat([x, x_new], 2)      

        Wl=self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]   

        return out[:, 0, :]        
