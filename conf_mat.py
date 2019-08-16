from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def conf_mat(true_list,predicted_list,img_cycle,train,test):
 
    cm = confusion_matrix(true_list, predicted_list)
    acc = accuracy_score(true_list, predicted_list)
    micro = f1_score(true_list, predicted_list, average='micro')   
    macro = f1_score(true_list, predicted_list, average='macro')
    
    labels = sorted(set(true_list).union(set(predicted_list)))
    
    cm_df = pd.DataFrame(cm,
                         index = labels , 
                         columns = labels)
    
    plt.figure(figsize=(10,7))
    sns_plot = sns.heatmap(cm_df, annot=True,cmap='Blues',fmt='g')
    plt.title('R2 - Accuracy is : '+ str(round(acc,2))+" Micro is : " +str(round(micro,2))+" Macro is : "+str(round(macro,2)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    figure = sns_plot.get_figure()  

    figure.savefig("Images/"+str(train)+str(test)+"/cycle_"+str(img_cycle)+".png",bbox_inches='tight')
    
    return acc
    