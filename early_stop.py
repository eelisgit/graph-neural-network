import numpy as np
import torch 
import conf_mat
class EarlyStopping:
    def __init__(self, patience=4):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.improve = False
   
    #def update(self, f1_score, model,true_list,predicted_list,img_cycle,train,test):
    def update(self, f1_score, enc_nn,metric_nn):

        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, enc_nn,metric_nn)
            self.improve = True
            
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            self.improve = False
            
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, enc_nn,metric_nn)
            self.counter = 0
            self.improve = True
 
    def save_checkpoint(self, f1_score, enc_nn,metric_nn):

        print(f'F1 score increased ({self.val_loss_min:.6f} --> {f1_score:.6f}).  Saving model ...')
        
        torch.save(enc_nn, 'best_enc_nn.t7')
        torch.save(metric_nn, 'best_metric_nn.t7')
        
        self.val_loss_min = f1_score
