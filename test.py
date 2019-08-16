import numpy as np
from utils import io_utils
from data import generator
from torch.autograd import Variable
import os
import pickle
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import conf_mat
import early_stop

def test_one_shot(e_stop,test_cycle,args, model, test_samples=5000, partition='test'):

    """Function used to perform testing (validation and final test)
    
    Parameters:
    e_stop (early_stop): early stop which monitors when we stop training
    test_cycle (int): current iteration cycle - used when creating confusion matrix file names
    args (Namespace): arguments from argparse
    model (list) : contains cnn, gnn and softmax models
    test_samples (int) : number of samples to test
    partition (string) : train, val or test 
    
    Returns:
    e_stop (early_stop) : we return the early_stop object which has been updated based upon the test
    
    """   
        
    io = io_utils.IOStream('checkpoints/' + args.exp_name + '/run.log')

    io.cprint('\n**** TESTING WITH %s ***' % (partition,))

    loader = generator.Generator(args.dataset_root, args, partition=partition, dataset=args.dataset)

    [enc_nn, metric_nn, softmax_module] = model
    enc_nn.eval()
    metric_nn.eval()
    correct = 0
    total = 0
    iterations = int(test_samples/args.batch_size_test)    
    
    true_list = []
    predicted_list = []
    
    with open(os.path.join('datasets', 'compacted_datasets', 'sensor_label_decoder.pickle'),
                  'rb') as handle:
        label_decoder = pickle.load(handle)
    
    sep = '\\'
    for temp in range(0,len(label_decoder)):
        label_decoder[temp] = label_decoder[temp].rsplit(sep, 1)[1]

    for i in range(iterations):
    
        data,labels_dict = loader.get_task_batch(batch_size=args.batch_size_test)
        [x, labels_x_cpu, _, x_global, xi_s, labels_yi_cpu] = data        
                                                                                                                                                                                                                                                           
        if args.cuda:
            xi_s = [batch_xi.cuda() for batch_xi in xi_s]
            labels_yi = [label_yi.cuda() for label_yi in labels_yi_cpu]
            x = x.cuda()
        else:
            labels_yi = labels_yi_cpu

        xi_s = [Variable(batch_xi) for batch_xi in xi_s]
        labels_yi = [Variable(label_yi) for label_yi in labels_yi]
        x = Variable(x)

        # Compute embedding from x and xi_s
        z = enc_nn(x)                                
        
        zi_s = [enc_nn(batch_xi) for batch_xi in xi_s]

        # Compute metric from embeddings
        output, out_logits = metric_nn(inputs=[z, zi_s, labels_yi])
        output = out_logits                               
        
        y_pred = softmax_module.forward(output)             

        y_pred = y_pred.data.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)                  
        labels_x_cpu = labels_x_cpu.numpy()
        labels_x_cpu = np.argmax(labels_x_cpu, axis=1)       
        
        for i in range(0,len(labels_x_cpu)):
            true_label = labels_dict[i,labels_x_cpu[i]]
            true_list.append(label_decoder[true_label])
            predicted_label = labels_dict[i,y_pred[i]]
            predicted_list.append(label_decoder[predicted_label])
            
        for row_i in range(y_pred.shape[0]):                
            if y_pred[row_i] == labels_x_cpu[row_i]:
                correct += 1
            total += 1

        if (i+1) % 100 == 0:
            io.cprint('{} correct from {} \tAccuracy: {:.3f}%)'.format(correct, total, 100.0*correct/total))
    acc = accuracy_score(true_list, predicted_list)
    micro = f1_score(true_list, predicted_list, average='weighted')   
    macro = f1_score(true_list, predicted_list, average='macro')
    
    e_stop.update(micro,enc_nn,metric_nn)
    
    if partition == 'test' or (partition == 'val' and e_stop.improve):     
        if partition == 'test':
            test_cycle = 999
        
        #Print confusion matrix
        conf_mat.conf_mat(true_list,predicted_list,test_cycle,args.train,args.test)

        test_labels = sorted(set(true_list).union(set(predicted_list)))
  
        print(classification_report(true_list, predicted_list, target_names=test_labels))
    
        print("Micro is " + str(micro))
    
    enc_nn.train()
    metric_nn.train()

    return e_stop
    
