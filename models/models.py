import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pickle
from models import gnn
import torchvision.models as models


class SensorModel(nn.Module):
    def __init__(self, args, emb_size):                       
        super(SensorModel, self).__init__()
        self.emb_size = emb_size
        self.ndf = 64
        self.args = args

        #Load pre-trained model
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(*list(vgg16.children())[:-2])

        #Freeze layers
        l_count = 0
        for param in self.vgg16.parameters():
            if l_count < 14:
                param.requires_grad = False
                l_count = l_count +1
        
        #Add fully connected layer
        self.fc2 = nn.Linear(25088, self.emb_size, bias=True)
        self.bn_fc2 = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):                                   
        """Function used to perform matrix-matrix multiplication on graph and edges.
        
        Parameters:
        input (list): List which contains edges and nodes of graph.
                
        Returns:
        Output (Tensor): Tensor of diemsnions [Batch_size x (Images +1) x Expansion size].
        
        """     
        x = self.vgg16(input)
        x = x.view(-1, 25088)                                     
        output = self.bn_fc2(self.fc2(x))                          

        return output


class MetricModel(nn.Module):
    def __init__(self, args, emb_size,train_loader):
        super(MetricModel, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = emb_size
        self.args = args
        
        #Create GNN
        if self.metric_network == 'gnn_iclr_nl':
            num_inputs = self.emb_size + train_loader.class_size  
            if self.args.dataset == 'sensor':
                self.gnn_obj = gnn.GNN_nl(args, num_inputs, nf=96, J=1,final_output=train_loader.class_size)

        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z, zi_s, labels_yi):
    
        """Function used for the GNN in order to calculate logits.
        
        Parameters:
        z (tensor): Tensor of dimensions [batch size x features] for unlabeled image
        zi_s (list): List containing labeled image data where each elements is a Tensor of dimensions [batch size x features]
        labels_yi (list): List containing image labels
  
                
        Returns:
        outputs (Tensor): Tensor of diemsnions [Batch_size x Number of classes].
        logits (Tensor): Tensor of diemsnions [Batch_size x Number of classes].
        
        """     
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))
        if self.args.cuda:
            zero_pad = zero_pad.cuda()
         
        labels_yi = [zero_pad] + labels_yi          
        zi_s = [z] + zi_s                            
 
        #Create nodes from features extracted via CNN
        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)] 
        nodes = [node.unsqueeze(1) for node in nodes]         
        nodes = torch.cat(nodes, 1)                           

        #Calculate logits
        logits = self.gnn_obj(nodes).squeeze(-1)  
        outputs = F.sigmoid(logits)
        
        return outputs, logits
        
    def forward(self, inputs):
    
        """Function used for the GNN in order to calculate logits.
        
        Parameters:
        inputs (list): list containing images in tensor format and the labels
        
        Returns:
        outputs (Tensor): Tensor of diemsnions [Batch_size x Number of classes].
        logits (Tensor): Tensor of diemsnions [Batch_size x Number of classes].

        """  
        
        [z, zi_s, labels_yi] = inputs

        if 'gnn_iclr' in self.metric_network:
            return self.gnn_iclr_forward(z, zi_s, labels_yi)
        else:
            raise NotImplementedError

class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
    
        """Function used to calculate log softmax
        
        Parameters:
        outputs (tensor): logits
        
        Returns:
        F.log_softmax(outputs) (Tensor): Log probability for each class

        """  
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise(NotImplementedError)


def load_model(model_name, args, io):

    """Function that tries to load a pre-existing model.
        
    Parameters:
    model_name (string): name of model
    args (Namespace): arguments from argparse
        
    Returns:
    model : Pre-trained model if it exists.

    """  
      
    try:
        model = torch.load('checkpoints/%s/models/%s.t7' % (args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None
    
def load_best_model(model_name, io):

    """Function that tries to load best pre-existing model.
        
    Parameters:
    model_name (string): name of model
        
    Returns:
    model : Best pre-trained model if it exists.

    """  
    try:
        model = torch.load('best_%s.t7' % (model_name))
        io.cprint('Loading Parameters from the best trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None


def create_models(args,train_loader):

    """Function that creates models when they do not exist.
        
    Parameters:
    args (Namespace): arguments from argparse
    train_loader (Generator): used to generate samples
        
    Returns:
    model : Best pre-trained model if it exists.

    """  
    
    if 'sensor' ==args.dataset:
        enc_nn = SensorModel(args, 128)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn, MetricModel(args, enc_nn.emb_size,train_loader)
