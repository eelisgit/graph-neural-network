from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image as pil_image
import pickle
from . import parser
import random

class Sensor(data.Dataset):
    def __init__(self, root, dataset='sensor'):
        self.root = root
        self.dataset = dataset
        self.seed = 10
        if not self._check_exists_():
            self._init_folders_()
            self._preprocess_()

    def _init_folders_(self):
    
        """Function used to initialise folders """   
        
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, 'sensor')):
            os.makedirs(os.path.join(self.root, 'sensor'))
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets')):
            os.makedirs(os.path.join(self.root, 'compacted_datasets'))

    def _check_exists_(self):
        """Function used to check data exists"""   
        if not os.path.exists(os.path.join(self.root, 'compacted_datasets', 'R1_train.pickle')) or not \
                os.path.exists(os.path.join(self.root, 'compacted_datasets', 'R1_test.pickle')):
            return False
        else:
            return True

    def _preprocess_(self):
        """Function used to pre-process data""" 
        print('\nPreprocessing Sensor images...')
        (class_names_train, images_path_train) = parser.get_image_paths(os.path.join(self.root, 'sensor','R1R2','R1R2','training'))   #11x9 = 99
          
        keys_all = sorted(list(set(class_names_train)))                      #11 classes
        label_encoder = {}                                                   #Maps label to integer
        label_decoder = {}                                                   #Maps integer to label
        for i in range(len(keys_all)):                                       #Create the two dictionary by looping 0-10 classes
            label_encoder[keys_all[i]] = i
            label_decoder[i] = keys_all[i]
   
        all_set = {}
        counter = 0
        for class_, path in zip(class_names_train, images_path_train):
            img = pil_image.open(path)                                       #Get image from path
            img = img.convert('RGB')
            img = img.resize((224, 224), pil_image.ANTIALIAS)
            img = np.array(img, dtype='float32')                             #Convert to array
            if label_encoder[class_] not in all_set:
                all_set[label_encoder[class_]] = []
            all_set[label_encoder[class_]].append(img)                       
            counter = counter +1                                             
       
        
        print("Number of images is " + str(counter))
        keys = sorted(list(all_set.keys()))                                                                             
        
        train_set = {}
        test_set = {}
        for i in range(11):
            print(i)
            train_set[keys[i]] = all_set[keys[i]]                   
        for i in range(11, len(keys)):
            test_set[keys[i]] = all_set[keys[i]]
            
        with open(os.path.join(self.root, 'compacted_datasets', 'R1_train.pickle'), 'wb') as handle:
            pickle.dump(train_set, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'R2_train.pickle'), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=2)

        with open(os.path.join(self.root, 'compacted_datasets', 'sensor_label_encoder.pickle'), 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted_datasets', 'sensor_label_decoder.pickle'), 'wb') as handle:
            pickle.dump(label_decoder, handle, protocol=2)
        
        print('Images preprocessed')

    def load_dataset(self, partition, args, size=(224, 224)):
    
        """Function used to load data

        Returns:
        data (dictionary): each dictionary key contains as a value each image for a specific class
        label_encoder (dictionary) : convert class names to int
        label_decoder (dictionary : convert int to class name
        """ 
        
        print("Loading dataset")
        print(partition)
        
        if partition == 'train':
            with open(os.path.join(self.root, 'compacted_datasets', args.train+'_'+partition+'.pickle'),
                          'rb') as handle:
                    data = pickle.load(handle)        
        else:
            with open(os.path.join(self.root, 'compacted_datasets', args.test+'_'+partition+'.pickle'),
                          'rb') as handle:
                    data = pickle.load(handle) 
                                                                                         
        with open(os.path.join(self.root, 'compacted_datasets', 'sensor_label_encoder.pickle'),
                  'rb') as handle:
            label_encoder = pickle.load(handle)
            
        with open(os.path.join(self.root, 'compacted_datasets', 'sensor_label_encoder.pickle'),
                  'rb') as handle:
            label_decoder = pickle.load(handle)

        # Resize images and normalize
        for class_ in data:
            for i in range(len(data[class_])):
                image_resized = np.transpose(data[class_][i], (2, 0, 1))                   
                data[class_][i] = image_resized

        print("Num classes " + str(len(data)))
        num_images = 0
        for class_ in data:
            num_images += len(data[class_])
        print("Num images " + str(num_images))
        
        return data, label_encoder,label_decoder
