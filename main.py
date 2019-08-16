#-------------------------------------------------------------------------------------------------------------------------------------------------
#This code builds upon the architecture created by Garcia et al.(2017) found in https://github.com/vgsatorras/few-shot-gnn with citation below:

#@article{garcia2017few,
#  title={Few-Shot Learning with Graph Neural Networks},
#  author={Garcia, Victor and Bruna, Joan},
#  journal={arXiv preprint arXiv:1711.04043},
#  year={2017}
#}
#-------------------------------------------------------------------------------------------------------------------------------------------------

from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data import generator
from utils import io_utils
import models.models as models
import test
import numpy as np
import pickle
import time
import early_stop

#Settings (can be overwritten in command prompt
parser = argparse.ArgumentParser(description='Few-Shot Learning with Graph Neural Networks')
parser.add_argument('--train', type=str, default='R1', metavar='N',
                    help='Dataset to train on')
parser.add_argument('--test', type=str, default='R2', metavar='N',
                    help='Dataset to test on')                    
parser.add_argument('--exp_name', type=str, default='debug_vx', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--batch_size', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--batch_size_test', type=int, default=10, metavar='batch_size',    
                    help='Size of batch)')
parser.add_argument('--decay_interval', type=int, default=10000, metavar='N',
                    help='Learning rate decay interval')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=5000, metavar='N',
                    help='how many batches between each model saving')
parser.add_argument('--test_interval', type=int, default=200, metavar='N',                   
                    help='how many batches between each test')
parser.add_argument('--metric_network', type=str, default='gnn_iclr_nl', metavar='N',
                    help='gnn_iclr_nl' + 'gnn_iclr_active')
parser.add_argument('--active_random', type=int, default=0, metavar='N',
                    help='random active ? ')
parser.add_argument('--dataset_root', type=str, default='datasets', metavar='N',
                    help='Root dataset')
parser.add_argument('--test_samples', type=int, default=50, metavar='N',
                    help='Number of shots')
parser.add_argument('--dataset', type=str, default='mini_imagenet', metavar='N',
                    help='omniglot')
parser.add_argument('--dec_lr', type=int, default=200, metavar='N',
                    help='Decreasing the learning rate every x iterations')
args = parser.parse_args()

def _init_():

    """Used for initializing folders and checking they exist. If not they will be created."""
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp models/models.py checkpoints' + '/' + args.exp_name + '/' + 'models.py.backup')
_init_()

#Stream used for logging data.
io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')
io.cprint(str(args))

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    io.cprint('Using GPU : ' + str(torch.cuda.current_device())+' from '+str(torch.cuda.device_count())+' devices')
    torch.cuda.manual_seed(args.seed)
else:
    io.cprint('Using CPU')

def train_batch(model, data):

    """Trains every batch by extracting features with CNN and then using GNN

    Parameters:
        model (list): List containing CNN, GNN and softamx models.
        data (list): List containing the data for the batch. This includes the unabeled sample and labeled samples.

    Returns:
        loss (Tensor): The loss for the batch after comparing predictions with true labels.
    """
    [enc_nn, metric_nn, softmax_module] = model
    [batch_x, label_x, batches_xi, labels_yi] = data
    
    #CNN
    z = enc_nn(batch_x)
    zi_s = [enc_nn(batch_xi) for batch_xi in batches_xi] 
    
    #GNN
    out_metric, out_logits = metric_nn(inputs=[z, zi_s, labels_yi])
    logsoft_prob = softmax_module.forward(out_logits)      

    # Loss
    label_x_numpy = label_x.cpu().data.numpy()              
    formatted_label_x = np.argmax(label_x_numpy, axis=1)    
    formatted_label_x = Variable(torch.LongTensor(formatted_label_x))
    if args.cuda:
        formatted_label_x = formatted_label_x.cuda()
    loss = F.nll_loss(logsoft_prob, formatted_label_x)  

    #Backwards
    loss.backward()          
    return loss

def train():

    """Main function used for training for model. Keeps iterating and updating parameters until early stop condition is reached."""

    #Generator is used to sample bacthes.
    train_loader = generator.Generator(args.dataset_root, args, partition='train', dataset=args.dataset)

    io.cprint('Batch size: '+str(args.batch_size))
    print("Learning rate is "+ str(args.lr))
    
    #Try to load models
    enc_nn = models.load_model('enc_nn', args, io)
    metric_nn = models.load_model('metric_nn', args, io)

    #creates models
    if enc_nn is None or metric_nn is None:
        enc_nn, metric_nn = models.create_models(args,train_loader)
                 
    softmax_module = models.SoftmaxModule()
    if args.cuda:
        enc_nn.cuda()
        metric_nn.cuda()

    io.cprint(str(enc_nn))   
    io.cprint(str(metric_nn))

    weight_decay = 0
    if args.dataset == 'sensor':  
        print('Weight decay '+str(1e-6))
        weight_decay = 1e-6

    opt_enc_nn = optim.Adam(filter(lambda p: p.requires_grad, enc_nn.parameters()),lr=args.lr, weight_decay=weight_decay)
    opt_metric_nn = optim.Adam(metric_nn.parameters(), lr=args.lr, weight_decay=weight_decay) 
   
    enc_nn.train()
    metric_nn.train()
    counter = 0
    total_loss = 0
    test_cycle = 0
    batch_idx = 0
  
    start = time.time()
    print("starting time count")
    e_stop = early_stop.EarlyStopping()
    
    #Start training loop
    while e_stop.early_stop is False:
        ####################
        # Train
        ####################
        #Load training batch
        data,_= train_loader.get_task_batch(batch_size=args.batch_size,cuda=args.cuda, variable=True)
        
        [batch_x, label_x, _, _, batches_xi, labels_yi] = data         
       
        opt_enc_nn.zero_grad()     
        opt_metric_nn.zero_grad()
        
        #Calculate loss
        loss_d_metric = train_batch(model=[enc_nn, metric_nn, softmax_module],
                                    data=[batch_x, label_x, batches_xi, labels_yi])
        #Update parameter
        opt_enc_nn.step()    
        opt_metric_nn.step()

        #Adjust learning rate
        adjust_learning_rate(optimizers=[opt_enc_nn, opt_metric_nn], lr=args.lr, iter=batch_idx)

        ####################
        # Display
        ####################
        counter += 1
        total_loss += loss_d_metric.item()
        if batch_idx % args.log_interval == 0:
                display_str = 'Train Iter: {}'.format(batch_idx)
                display_str += '\tLoss_d_metric: {:.6f}'.format(total_loss/counter)
                io.cprint(display_str)
                counter = 0
                total_loss = 0

        ####################
        # Test
        ####################
        #Testing at specific itnervals
        if (batch_idx + 1) % args.test_interval == 0 or batch_idx == 0:       
            if batch_idx == 20:
                test_samples = 200
            else:
                test_samples = 300
                
            e_stop = test.test_one_shot(e_stop,test_cycle,args, model=[enc_nn, metric_nn, softmax_module],
                                              test_samples=test_samples, partition='val')              
                                      
            enc_nn.train()        
            metric_nn.train()
                
            test_cycle = test_cycle+1
            
            end = time.time()
            io.cprint("Time elapsed : " + str(end - start))
            print("Time elapsed : " + str(end - start))
            
        ####################
        # Save model
        ####################
        #Save model at specific interval
        if (batch_idx + 1) % args.save_interval == 0:
            torch.save(enc_nn, 'checkpoints/%s/models/enc_nn.t7' % args.exp_name)
            torch.save(metric_nn, 'checkpoints/%s/models/metric_nn.t7' % args.exp_name)
            
        batch_idx = batch_idx +1
        
    #Test after training
    #Load best model
    final_enc_nn = models.load_best_model('enc_nn', io)
    final_metric_nn = models.load_best_model('metric_nn', io)
    
    final_enc_nn.cuda()
    final_metric_nn.cuda()
    
    test.test_one_shot(e_stop,test_cycle,args, model=[final_enc_nn, final_metric_nn, softmax_module],
                       test_samples=args.test_samples,partition='test')

def adjust_learning_rate(optimizers, lr, iter):

    """Function used for reducing the learning rate.
    
    Parameters:
        optimizers (list): List containing the optimizers.
        lr (float): Current learning rate of the optimizer.
        iter : The current iteration of trainig cycle.

    """
    new_lr = lr * (0.5**(int(iter/args.dec_lr)))
    
    if iter % args.log_interval == 0:
        print("Learning rate is " + str(new_lr))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

if __name__ == "__main__":
    train()

