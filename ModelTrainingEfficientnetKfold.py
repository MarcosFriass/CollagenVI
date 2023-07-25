#!/usr/bin/env python

# Basic system routines packages
import os, sys

print(sys.version)

# Keras imports
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.applications import *
from tensorflow.keras.utils import Sequence
from tensorflow.python.client import device_lib
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.models import load_weights


#Set GPU
print('-------------------------------')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())
print('-------------------------------')

# Others
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, \
    PrecisionRecallDisplay, confusion_matrix, classification_report
import numpy as np
import shutil
import tiler
import cv2
import seaborn as sns
import datetime
import albumentations as A
from albumentations.augmentations.transforms import *

# Add the Utils folder path to the sys.path list
sys.path.append('../Tools/')

#set a seed for reproducibility
seed = 41
np.random.seed(seed)

# Import model custom metrics
from CustomModelMetrics import m_recall
from CustomModelMetrics import m_f1
from CustomModelMetrics import m_precision

#Methods
def usage():
    print("Usage Examples:")
    print("python ModelTrainingEfficientnet.py --help")
    print("python ModelTrainingEfficientnet.py --dataDir='dataFile' --imgDir='imgDirPath' --imgsize='imageSize'")
    print("python ModelTrainingEfficientnet.py --dataDir='dataFile' --imgDir='imgDirPath' --imgsize='imageSize' --epochs=100 --batch_size=32 --network='efficientnet'")
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description = "Script to train the CNN model", epilog = "")
    parser.add_argument("-d", "--dataDir", default='GT_pediatricos.csv',required=True,help="Path to the directory containing the GT anotation (REQUIRED).",dest="dataDir")
    parser.add_argument("-img", "--imgDir", default='imgDIR',required=True,help="Path to the directory containing the TIFF images (REQUIRED).", dest="imgDir")
    parser.add_argument("-s", "--imgsize", default=224, required=True, help="Patch dimension (REQUIRED). For Efficientnet use 224", dest="imgsize")
    parser.add_argument("-e", "--epochs", default=100, required=False, help="Number of training epochs (OPTIONAL).", dest="epochs")
    parser.add_argument("-bs", "--batch_size", default=64, required=False, help="Training batch size (OPTIONAL).", dest="batch_size")
    parser.add_argument("-n", "--network", default='efficientnet', required=False, help="Network to train Efficientnet. It can be efficientnet or efficientnet_tl (OPTIONAL)",dest="network")
    parser.add_argument("-rp", "--rebuild_patches", default=False, required=False, help="Rebuild patches (OPTIONAL)", dest="rebuild_patches")
    parser.add_argument("-o", "--overlap", default=0, required=False, help="Overlap between patches (OPTIONAL)", dest="overlap")
    parser.add_argument("-u", "--usage", help="Usage examples", dest="usage", action = 'store_true')
    return parser.parse_args()

def create_model(img_width, img_height, network):
    # Use the correct Keras input shape
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    NUM_CLASSES = 2

    #Use Efficientnet in a transfer learning mode
    if network == 'efficientnet_tl':
        inputs = Input(shape=input_shape)

        baseModel = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

        # Freeze the pretrained weights
        baseModel.trainable = False

        # Rebuild top
        x = GlobalAveragePooling2D(name="avg_pool")(baseModel.output)
        outputs = Dense(NUM_CLASSES, activation="softmax")(x)

        # Compile
        model = Model(inputs, outputs, name="EfficientNet")

        optimizer = "adam"

    # debug commands to stop code

    # Use Efficientnet in a fine-tuning mode
    if network == 'efficientnet':
        inputs = Input(shape=input_shape)

        baseModel = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

        # Freeze the pretrained weights
        baseModel.trainable = True

        # Rebuild top
        x = GlobalAveragePooling2D(name="avg_pool")(baseModel.output)
        outputs = Dense(NUM_CLASSES, activation="softmax")(x)

        

        # Compile
        model = Model(inputs, outputs, name="EfficientNet")

        optimizer = "adam"

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', m_recall, m_precision, m_f1])
    # model.summary()

    return model

#Class to create data aumentation
class AugmentDataGenerator(Sequence):
    def __init__(self, datagen, augment=None):
        self.datagen = datagen
        if augment is None:
            self.augment = A.Compose([])
        else:
            self.augment = augment

    def __len__(self):
        return len(self.datagen)

    def __getitem__(self, x):
        images, *rest = self.datagen[x]
        augmented = []
        for image in images:
            image = self.augment(image=image)['image']
            augmented.append(image)
        return (np.array(augmented), *rest)

#Scheduler
def lr_time_based_decay(epoch, lr):
    initial_learning_rate = 0.001
    epochs = 20
    decay = initial_learning_rate / epochs
    return lr * 1 / (1 + decay * epoch)


# SCRIPT
if __name__ == '__main__':

    args = parse_args()
    if(args.usage):
        usage()

    # Run options and parameters
    data_csv = args.dataDir
    image_dir = args.imgDir
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    image_size_list = args.imgsize.strip('][').split(',')
    network = str(args.network)
    rebuild_patches = bool(int(args.rebuild_patches))
    overlap_list = args.overlap.strip('][').split(',')



    for image_size in image_size_list:
        for overlap in overlap_list:

            image_size = int(image_size)
            overlap = int(overlap)

            # Our images dimension
            img_width, img_height = image_size, image_size

            # Augmentation configuration for the training and validation sets + rescaling
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            horizontal_flip=False, 
                                            vertical_flip=False, 
                                            rotation_range=0,
                                            fill_mode='nearest')
            # val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
            

            #Read Gt anotations
            data = pd.read_csv(data_csv)
            Y = data[['labels']]
            #Remove doubtful patient cases. Subject with an accuracy below 24%
            acc_file = r"accuracyBySubject.csv"
            accuraciesSubjects = pd.read_csv(acc_file)
            control_df = accuraciesSubjects.iloc[:21, :]
            patients_df = accuraciesSubjects.iloc[21:, :]
            patients_df_sorted = patients_df.sort_values('accuracy')
            patients_df_filtered = patients_df_sorted.loc[patients_df_sorted['accuracy'] > 24, :]

            #Set library to create patches
            n = 411 #GT examples
            overlap = int(image_size*overlap/100)

            #Create results folder
            results_path = 'Results_e' + str(epochs) + '_s' + str(image_size) +'_b' + str(batch_size) + '_o' + str(overlap) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_Kfold'
            #check if the folder exists
            if not os.path.exists(results_path):
                os.mkdir(results_path)

            # Save best weights found so far (checkpoint) and early stopping
            bestWeightsPath = os.path.join(results_path, "Model_Weights_test_" + str(epochs) + str(image_size) + str(batch_size) + ".h5")
            checkpoint = callbacks.ModelCheckpoint(bestWeightsPath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
            early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max')
            LearningRateScheduler = callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)

            #Create log dir for Tensorboard
            log_dir = os.path.join(results_path, "log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True, update_freq='epoch')
            callbacks_list = [LearningRateScheduler,checkpoint, early_stop,tensorboard]


            tiler_object = tiler.Tiler(data_shape=(1024, 1024, 3), tile_shape=(image_size, image_size, 3), channel_dimension=2,overlap=overlap)
            with tf.device('/device:GPU:0'):
                patchesDirectory = 'Patches_size' + str(image_size) + '_overlap' + str(overlap)
                if rebuild_patches:
                    #check if the folder exists and if it does, remove it
                    if os.path.isdir(patchesDirectory):
                        shutil.rmtree(patchesDirectory)
                    if not os.path.isdir(patchesDirectory):
                        os.mkdir(patchesDirectory)
                    
                    for i in range(n):
                        img = cv2.imread(data.loc[:, 'name'][i]+'f')
                        for tile_id, tile in tiler_object(img):
                            p = tiler_object.get_tile(img, tile_id)
                            cv2.imwrite(os.path.join(patchesDirectory, data.loc[:, 'name'][i].split(".tif")[0][-23:] + '_patch_' + str(tile_id)) + '.tiff', p)



                #Split train-test
                #training_data, validation_data = train_test_split(data, test_size=0.2)
                #generate a vector with an index identifier for each subject, extracted from data['name'] value as the third element of the path when splitted by '/' and then the three first elements when splitted by '_'
                data['subject'] = data['name'].apply(lambda x: str(x.split('/')[2].split('_')[:3]))

                # geenerate a vector with the labels for each subject
                data['labels'] = data['name'].apply(lambda x: 0 if str(x.split('/')[1]) == 'CONTROL' else 1)


                unique_subjects = data['subject'].unique()
                unique_labels = []
                for u in range(len(unique_subjects)):
                    ID = unique_subjects[u]
                    #find the index of the first element of the vector data['subject'] that matches with the ID
                    index = data[data['subject'] == ID].index[0]
                    #append the label of the subject to the vector unique_labels
                    unique_labels.append(data['labels'][index])
                
            
                    

                #create a 41 number vector call indexes_shuffle
                indexes_shuffle = np.arange(41)
                #shuffle the vector
                np.random.shuffle(indexes_shuffle)

                #use the vector to shuffle the unique_subjects vector
                unique_subjects = unique_subjects[indexes_shuffle]
                #same for unique_labels
                unique_labels = np.array(unique_labels)[indexes_shuffle]

                # use straified k-fold to split the data in train and test using the unique_subjects vector and the unique_labels vector as input to perform a 5-fold stratified split
                skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
                skf.get_n_splits(unique_subjects, unique_labels)
                K_iter = 0
                test_accuracy_vector = []
                AUC_vector = []
                cm_global = np.zeros((2,2)).astype(int)
                cm_global_biopsy = np.zeros((2,2)).astype(int)

                for train_index, test_index in skf.split(unique_subjects, unique_labels):
                    K_iter += 1
                    print("-------------------- K-fold iteration: ", K_iter, " ----------------------")

                    train_subjects, test_subjects = unique_subjects[train_index], unique_subjects[test_index]

                # #split the unique_subjects vector in two vectors, one for training and one for validation using a stratification method
                # train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, stratify=unique_labels)

                #split the train_subjects vector in two vectors, one for training and one for validation using a stratification method
                    # unique_labels2 = []
                    # for u in range(len(train_subjects)):
                    #     ID = train_subjects[u]
                    #     #find the index of the first element of the vector data['subject'] that matches with the ID
                    #     index = data[data['subject'] == ID].index[0]
                    #     #append the label of the subject to the vector unique_labels
                    #     unique_labels2.append(data['labels'][index])
        
                

                    # train_subjects, validation_subjects = train_test_split(train_subjects, test_size=0.1, stratify=unique_labels2)

                #create a new vector with the index identifier for each subject
                    training_data = data[data['subject'].isin(train_subjects)]
                    # validation_data = data[data['subject'].isin(validation_subjects)]
                    test_data = data[data['subject'].isin(test_subjects)]

                # #from data['subject'] vector create a new numerical vector with the index of each subject
                # data['subject_index'] = pd.factorize(data['subject'])[0]


                # #Split train-test using stratified k-fold from sklearn
                # skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
                # skf.get_n_splits(data['name'].values, data['labels'].values)
                # for train_index, test_index in skf.split(data['name'].values, data['labels'].values, data['subject_index'].values):
                #     training_data, validation_data = data.iloc[train_index], data.iloc[test_index]
                #     break
                
                #save training and validation data
                    training_data.to_csv(os.path.join(results_path, "training_data_"+ str(K_iter)+ ".csv"))
                    # validation_data.to_csv(os.path.join(results_path, "validation_data_"+ str(K_iter)+ ".csv"))
                    test_data.to_csv(os.path.join(results_path, "test_data_"+ str(K_iter)+ ".csv"))



                #debugging commands to stop code


                # #Take into account that depending on the OS, the paths can be different. This code can be runned in Linux.
                # for t in training_data['name'].values:
                #     if str(t[8:15]) != 'CONTROL':
                #             if not str(t.split('/')[2][:6]) in str(patients_df_filtered['name'].values):
                #                 training_data = training_data.loc[training_data['name'] != t, :]

                # for v in validation_data['name'].values:
                #     if str(v[8:15]) != 'CONTROL':
                #             if not str(v.split('/')[2][:6]) in str(patients_df_filtered['name'].values):
                #                 validation_data = validation_data.loc[validation_data['name'] != v, :]

                #debugging commands to stop code




                #Create train-test directories to save patches. Format: patches+epochs+patch_size+batch_size+network_used+ timestamp
                    #check if the folder exists and if it does, remove it



                    outputDirectory = 'patches_e' + str(epochs) + '_s' + str(image_size) + '_b' + str(batch_size) +'_o' +str(overlap)
                    outputTrainDirectory = outputDirectory +'/Train'
                    outputTrainDirectoryControl = outputDirectory + '/Train/CONTROL'
                    outputTrainDirectoryPatient = outputDirectory+ '/Train/PACIENTES'
                    outputValDirectory = outputDirectory+ '/Validation'
                    outputValDirectoryControl = outputDirectory+ '/Validation/CONTROL'
                    outputValDirectoryPatient = outputDirectory+ '/Validation/PACIENTES'
                    outputTestDirectory = outputDirectory+ '/Test'
                    outputTestDirectoryControl = outputDirectory+ '/Test/CONTROL'
                    outputTestDirectoryPatient = outputDirectory+ '/Test/PACIENTES'

                    #check if the folder outputDirectory exists and if it does, remove it
                    if os.path.isdir(outputDirectory):
                        shutil.rmtree(outputDirectory)
                        redistribution=True


                    redistribution=False
                    if not os.path.isdir(outputDirectory):
                        redistribution = True
                        os.mkdir(outputDirectory)
                    if not os.path.isdir(outputTrainDirectory):
                        os.mkdir(outputTrainDirectory)
                    if not os.path.isdir(outputTrainDirectoryControl):
                        os.mkdir(outputTrainDirectoryControl)
                    if not os.path.isdir(outputTrainDirectoryPatient):
                        os.mkdir(outputTrainDirectoryPatient)
                    if not os.path.isdir(outputValDirectory):
                        os.mkdir(outputValDirectory)
                    if not os.path.isdir(outputValDirectoryControl):
                        os.mkdir(outputValDirectoryControl)
                    if not os.path.isdir(outputValDirectoryPatient):
                        os.mkdir(outputValDirectoryPatient)
                    if not os.path.isdir(outputTestDirectory):
                        os.mkdir(outputTestDirectory)
                    if not os.path.isdir(outputTestDirectoryControl):
                        os.mkdir(outputTestDirectoryControl)
                    if not os.path.isdir(outputTestDirectoryPatient):
                        os.mkdir(outputTestDirectoryPatient)

                    if redistribution:
                        #according to the training and validation data, move the pertinent patches from the patchesDirectory to the train and validation directories
                        for patch in os.listdir(patchesDirectory):            
                            # import pdb; pdb.set_trace()
                            if str(patch.split('_')[0:3]) in training_data['subject'].values:
                                #search the index of the subject in the training_data vector
                                index = np.where(training_data['subject'].values == str(patch.split('_')[0:3]))[0][0]
                                #if the label of the subject is 0, move the patch to the control directory
                                if training_data['labels'].values[index] == 0:
                                    shutil.copy(os.path.join(patchesDirectory, patch), outputTrainDirectoryControl)
                                #if the label of the subject is 1, move the patch to the patient directory
                                else:
                                    shutil.copy(os.path.join(patchesDirectory, patch), outputTrainDirectoryPatient)
                            # elif str(patch.split('_')[0:3]) in validation_data['subject'].values:
                            #     #search the index of the subject in the validation_data vector
                            #     index = np.where(validation_data['subject'].values == str(patch.split('_')[0:3]))[0][0]
                            #     #if the label of the subject is 0, move the patch to the control directory
                            #     if validation_data['labels'].values[index] == 0:
                            #         shutil.copy(os.path.join(patchesDirectory, patch), outputValDirectoryControl)
                            #     #if the label of the subject is 1, move the patch to the patient directory
                            #     else:
                            #         shutil.copy(os.path.join(patchesDirectory, patch), outputValDirectoryPatient)
                            elif str(patch.split('_')[0:3]) in test_data['subject'].values:
                                #search the index of the subject in the test_data vector
                                index = np.where(test_data['subject'].values == str(patch.split('_')[0:3]))[0][0]
                                #if the label of the subject is 0, move the patch to the control directory
                                if test_data['labels'].values[index] == 0:
                                    shutil.copy(os.path.join(patchesDirectory, patch), outputTestDirectoryControl)
                                #if the label of the subject is 1, move the patch to the patient directory
                                else:
                                    shutil.copy(os.path.join(patchesDirectory, patch), outputTestDirectoryPatient)


                # #Save patches
                # for i in range(n):
                #     try:
                #         label = training_data.loc[:, 'labels'][i]
                #         img = cv2.imread(training_data.loc[:, 'name'][i]+'f')
                #         if label == 0:
                #             for tile_id, tile in tiler(img):
                #                 p = tiler.get_tile(img, tile_id)
                #                 cv2.imwrite(os.path.join(outputTrainDirectoryControl,
                #                                          training_data.loc[:, 'name'][i].split(".tif")[0][
                #                                          -23:] + '_patch_' + str(tile_id)) + '.tiff', p)
                #         else:
                #             for tile_id, tile in tiler(img):
                #                 p = tiler.get_tile(img, tile_id)
                #                 cv2.imwrite(os.path.join(outputTrainDirectoryPatient,
                #                                          training_data.loc[:, 'name'][i].split(".tif")[0][
                #                                          -23:] + '_patch_' + str(tile_id)) + '.tiff', p)
                #     except:
                #         print("The image belongs to validation set")

                # for i in range(n):
                #     try:
                #         label = validation_data.loc[:, 'labels'][i]
                #         img = cv2.imread(validation_data.loc[:, 'name'][i]+'f')
                #         if label == 0:
                #             for tile_id, tile in tiler(img):
                #                 p = tiler.get_tile(img, tile_id)
                #                 cv2.imwrite(os.path.join(outputTestDirectoryControl,
                #                                          validation_data.loc[:, 'name'][i].split(".tif")[0][
                #                                          -23:] + '_patch_' + str(tile_id)) + '.tiff', p)
                #         else:
                #             for tile_id, tile in tiler(img):
                #                 p = tiler.get_tile(img, tile_id)
                #                 cv2.imwrite(os.path.join(outputTestDirectoryPatient,
                #                                          validation_data.loc[:, 'name'][i].split(".tif")[0][
                #                                          -23:] + '_patch_' + str(tile_id)) + '.tiff', p)
                #     except:
                #         print("The image belongs to train set")

                #Set generators. Validation and test generators work over the same dataset but with a different setting
                    train_generator = train_datagen.flow_from_directory(outputTrainDirectory,
                                                                        target_size=(img_height, img_width),
                                                                        batch_size=batch_size, shuffle=True)
                    #train_generator = AugmentDataGenerator(train_generator, A.Compose([A.RandomRotate90(p=1.0)]))


                    # validation_generator = val_datagen.flow_from_directory(outputValDirectory,
                    #                                                         target_size=(img_height, img_width),
                    #                                                         batch_size=batch_size,shuffle=True)

                    test_generator = test_datagen.flow_from_directory(outputTestDirectory,
                                                                    target_size=(img_height, img_width),
                                                                    class_mode=None, batch_size=1,shuffle=False)

                # Build the model
                    model = create_model(img_width, img_height, network)#, training=True)

                # Generate a print
                    print('------------------------------------------------------------------------')

                # debug commands to stop code
                #import pdb; pdb.set_trace()

                # Save the model training history as CSV
                    # history = model.fit(train_generator, batch_size=batch_size, epochs=epochs,
                    #                     validation_data=validation_generator, callbacks=callbacks_list)
                    history = model.fit(train_generator, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)
                    pd.DataFrame(history.history).to_csv(os.path.join(results_path,
                                                                    "ModelTraining_" + str(epochs) + str(image_size) + str(
                                                                        batch_size) + '_' + str(K_iter) + ".csv"))

                # #save the model weights
                    model.save_weights(os.path.join(results_path, "Model_Weights_" + str(epochs) + str(image_size) + str(batch_size) + '_' + network + ".h5"))
                # model.load_weights("Results_e1_s224_b32/Model_Weights_122432_efficientnet.h5")#, custom_objects={'m_recall': m_recall, 'm_precision': m_precision,'m_f1': m_f1 })


                #debug

                # Generate generalization metrics
                    # scores = model.evaluate(validation_generator, verbose=1)
                    # print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

                #Make predictions over test generator
                    test_generator.reset()
                    predictions_test = model.predict(test_generator, verbose=1)
                    classes = test_generator.classes[test_generator.index_array]
                    pred = np.argmax(predictions_test, axis=1)

                # read the folder outputTestDirectory and save the name of each file in a vector, additionally, save the labels of each file in a vector, being a 0 if the file belongs to the control group and a 1 if the file belongs to the patient group

                    files = []
                    for r, d, f in os.walk(outputTestDirectory):
                        for file in f:
                            files.append(os.path.join(r, file))
                    files.sort()

                # create a vector with the name of each file without the extension
                    files_names = []
                    for i in range(len(files)):
                        files_names.append(files[i].split('/')[-1].split('.')[0])

                    # split by \ and take the first five elements together in a single string with _ as separator
                    files_names_ids = []
                    for i in range(len(files_names)):
                        files_names_ids.append('_'.join(files_names[i].split('_')[:5]))

                    #unique files_names_ids but maintaining the order
                    unique_files_names_ids = []
                    for i in range(len(files_names_ids)):
                        if files_names_ids[i] not in unique_files_names_ids:
                            unique_files_names_ids.append(files_names_ids[i])

                    # perform a majority voting of the prediction of each patch of the same image
                    pred_majority = []
                    for i in range(len(unique_files_names_ids)):
                        # find the indexes of the patches of the same image
                        indexes = np.where(np.array(files_names_ids) == unique_files_names_ids[i])[0]
                        # find the predictions of the patches of the same image
                        predictions = pred[indexes]
                        # find the most frequent prediction
                        pred_majority.append(np.bincount(predictions).argmax())
                    
                    biopsy_ids= []
                    for i in range(len(files_names)):
                        biopsy_ids.append('_'.join(files_names[i].split('_')[:3]))


                    #unique biopsy_ids_unique but maintaining the order   
                    biopsy_ids_unique = []
                    for i in range(len(biopsy_ids)):
                        if biopsy_ids[i] not in biopsy_ids_unique:
                            biopsy_ids_unique.append(biopsy_ids[i])

                    # create a vector with the label of each biopsy_ids_unique element, a 0 if the biopsy belongs to the control group and a 1 if the biopsy belongs to the patient group
                    GT_biopsy = []
                    for i in range(len(biopsy_ids_unique)):
                        # find the indexes of the patches of the same biopsy
                        indexes = np.where(np.array(biopsy_ids) == biopsy_ids_unique[i])[0]
                        # find the labels of the patches of the same biopsy
                        labels = classes[indexes]
                        # find the most frequent label
                        GT_biopsy.append(np.bincount(labels).argmax())

                    #perform a majority voting of the prediction of each biopsy
                    pred_majority_biopsy = []
                    for i in range(len(biopsy_ids_unique)):
                        #find the indexes of the patches of the same biopsy
                        indexes = np.where(np.array(biopsy_ids) == biopsy_ids_unique[i])[0]
                        #find the predictions of the patches of the same biopsy
                        predictions = pred[indexes]
                        #find the most frequent prediction
                        pred_majority_biopsy.append(np.bincount(predictions).argmax())


                    # perform confusion matrix comparing the predictions of the biopsy with the GT
                    cm_biopsy = confusion_matrix(GT_biopsy, pred_majority_biopsy)
                    plt.figure()
                    plot = sns.heatmap(cm_biopsy, annot=True, fmt='d')
                    plot.set(xlabel='Predicted', ylabel='True')
                    plot.set_title('Confusion matrix')
                    plt.savefig(os.path.join(results_path, "Confusion matrix biopsy " + str(epochs) + str(image_size) + str(batch_size) + str(K_iter)+".png"))
                    plt.close()

                    cm_global_biopsy += cm_biopsy

                    plt.figure()
                    plot = sns.heatmap(cm_global_biopsy, annot=True, fmt='d')
                    plot.set(xlabel='Predicted', ylabel='True')
                    plot.set_title('Confusion matrix')
                    plt.savefig(os.path.join(results_path, "Confusion matrix biopsy " + str(epochs) + str(image_size) + str(batch_size) + "GLOBAL.png"))
                    plt.close()

                    print(pred_majority_biopsy)



                # agroup classes coming from the same id

                    # Plot Classification Report and Confusion matrix
                    print('CLASSIFICATION REPORT:')
                    print(classification_report(classes, pred))

                    # print accuracy from testing predictions
                    print('ACCURACY:')
                    accuracy = np.sum(pred == classes) / len(classes)

                    test_accuracy_vector.append(accuracy)
                    print(accuracy)


                    #Plot confusion matrix
                    cm = confusion_matrix(classes, pred)

                    plt.figure()
                    plot = sns.heatmap(cm, annot=True, fmt='d')
                    plot.set(xlabel='Predicted', ylabel='True')
                    plot.set_title('Confusion matrix')
                    plt.savefig(os.path.join(results_path, "Confusion matrix " + str(epochs) + str(image_size) + str(batch_size) + str(K_iter)+".png"))
                    plt.close()

                    cm_global += cm

                    plt.figure()
                    plot = sns.heatmap(cm_global, annot=True, fmt='d')
                    plot.set(xlabel='Predicted', ylabel='True')
                    plot.set_title('Confusion matrix')
                    plt.savefig(os.path.join(results_path, "Confusion matrix " + str(epochs) + str(image_size) + str(batch_size) + "GLOBAL.png"))
                    plt.close()

                #Plot ROC curve
                # plt.figure()
                    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_generator.classes, np.array(predictions_test)[:, 1])
                    auc_keras = auc(fpr_keras, tpr_keras)
                    AUC_vector.append(auc_keras)

                # plt.plot(fpr_keras, tpr_keras, marker='.', label='Neural Network (auc = %0.2f)' % auc_keras)
                    print('AUC= ', auc_keras)
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('Receiver operating characteristic')
                # plt.legend(loc="lower right")
                # plt.savefig(os.path.join(results_path, "Receiver operating characteristic " + str(epochs) + str(image_size) + str(batch_size) + ".png"))
                # plt.close()

                # # Plot Precision-Recall curve
                # plt.figure()
                # precision, recall, thresholds = precision_recall_curve(test_generator.classes, np.array(predictions_test)[:,1])
                # auc_PR = auc(recall, precision)
                # plt.plot(recall, precision, marker='.', label='Neural Network (auc = %0.2f)' % auc_PR)
                # plt.xlabel('Recall')
                # plt.ylabel('Precision')
                # plt.title('Precision-Recall curve')
                # plt.legend(loc="lower right")
                # plt.savefig(os.path.join(results_path, "Precision-Recall curve " + str(epochs) + str(image_size) + str(
                #     batch_size) + ".png"))
                # plt.close()

                    # summarize history for accuracy
                    plt.figure()
                    plt.plot(history.history['accuracy'])
                    plt.title('Model Accuracy')
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')
                    plt.legend(['train'], loc='upper left')
                    plt.savefig(os.path.join(results_path, "Model Accuracy " + str(epochs) + str(image_size) + str(batch_size) + str(K_iter)+".png"))
                    plt.close()

                    # summarize history for loss
                    plt.figure()
                    plt.plot(history.history['loss'])
                    plt.title('Model Loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train'], loc='upper left')
                    plt.savefig(os.path.join(results_path, "Model Loss " + str(epochs) + str(image_size) + str(batch_size) + str(K_iter)+".png"))
                    plt.close()

                # # Serialize model to JSON
                # model_json = model.to_json()
                # with open(os.path.join(results_path,"ModelTrainingEfficientnet.json"), "w") as json_file:
                #     json_file.write(model_json)

                #shutil.rmtree(outputDirectory)
            
            print('--------------------------RESULTS--------------------------')
            print("Vector of test accuracy: ", test_accuracy_vector)
            print("Vector of AUC: ", AUC_vector)

            print("Average Accuracy 5-Fold CrossValidation " + str(np.mean(test_accuracy_vector)))
            print("Average AUC 5-Fold CrossValidation " + str(np.mean(AUC_vector)))

            #save these results in a txt file
            with open(os.path.join(results_path, "Results.txt"), "w") as text_file:
                print("Vector of test accuracy: ", test_accuracy_vector, file=text_file)
                print("Vector of AUC: ", AUC_vector, file=text_file)
                print("Average Accuracy 5-Fold CrossValidation " + str(np.mean(test_accuracy_vector)), file=text_file)
                print("Average AUC 5-Fold CrossValidation " + str(np.mean(AUC_vector)), file=text_file)

            
            # obtain total value of cm_global
            total = np.sum(cm_global)
            # create a new matrix with the percentages of each value of cm_global
            cm_global_percentages = np.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    cm_global_percentages[i,j] = cm_global[i,j]/total
            
            # Plot confusion matrix
            plt.figure()
            plot = sns.heatmap(cm_global_percentages, annot=True, fmt='.2%', cmap='Blues')
            plot.set(xlabel='Predicted', ylabel='True')
            plot.set_title('Confusion matrix')
            plt.savefig(os.path.join(results_path, "Confusion matrix " + str(epochs) + str(image_size) + str(batch_size) + "GLOBAL_PERCENTAGES.png"))
            plt.close()

            # obtain total value of cm_global_biopsy
            total_biopsy = np.sum(cm_global_biopsy)
            # create a new matrix with the percentages of each value of cm_global_biopsy
            cm_global_biopsy_percentages = np.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    cm_global_biopsy_percentages[i,j] = cm_global_biopsy[i,j]/total_biopsy

            # Plot confusion matrix
            plt.figure()
            plot = sns.heatmap(cm_global_biopsy_percentages, annot=True, fmt='.2%', cmap='Blues')
            plot.set(xlabel='Predicted', ylabel='True')
            plot.set_title('Confusion matrix')
            plt.savefig(os.path.join(results_path, "Confusion matrix biopsy " + str(epochs) + str(image_size) + str(batch_size) + "GLOBAL_PERCENTAGES.png"))
            plt.close()

            




