"""
this script containts the one shot kill poison attack
"""
import numpy as np
from util_one_shot_kill_attack import *
from scipy import misc
import os

#location of dog and fish directories
dataDir = '/home/tdteach/data/dogfish/'
dogDir = dataDir+'main_dog/'
fishDir = dataDir+'fish/'
finalDir = dataDir+'final_images/'

#parameters
firsTime = False #flag that if true, will load the raw images from the directories and extract the features and prepare the inputs for training.
startFromClosest = True #if ture, for every target, we would select the base to be the closest instance from the other class in the test set. Closeness is measured by L2 distance in feature space reps
threshold = 3.5 #threshold for L2 distance in feature space


#if startFromClosest == False:
#these are some preferred images as base instances which will be used for generating poison instances for the attacks
#class_dog_base_id_in_test = [507, 406, 493]
#class_fish_base_id_in_test = [738,718,991,962]
#class_dog_base_id_in_test = [507, 406, 493]
#class_fish_base_id_in_test = [738,718,991,962]

if firsTime:
    #load the images into numpy arrays
    allDogs = load_images_from_directory('dog', dogDir)
    allFishes = load_images_from_directory('fish', fishDir)

    #remove the ones that might cause an issue with inception - these are the ones which have less than 3 dimensions
    allDogs = clean_data(X=allDogs)
    allFishes = clean_data(X=allFishes)

    #save the images to file for future use
    directory = finalDir
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'dogInput.npy',allDogs)
    directory = finalDir
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'fishInput.npy',allFishes)

    allDogs = np.load(finalDir+'dogInput.npy')
    allFishes = np.load(finalDir+'fishInput.npy')
    #get the feature representations and save them
    dogFeats = get_feat_reps(X=allDogs, class_t='dog')
    directory = finalDir
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'dogFeats.npy',dogFeats)
    fishFeats = get_feat_reps(X=allFishes, class_t='fish')
    directory = finalDir
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory+'fishFeats.npy',fishFeats)

    # load the bottleneck tensors and do the train test split and save it
    # X_tr is X_tr_feats
    # X_test is X_tst_feats
    # X_inp_test is X_tr_inp and also the training part of allDogs
    X_tr, X_test, X_inp_tr, X_inp_test, Y_tr, Y_test = load_bottleNeckTensor_data(directory=finalDir,saveEm=True)



#load the training and test data
directorySaving = dataDir+'XY/'
all_datas = ['X_tr_feats', 'X_tst_feats', 'X_tr_inp', 'X_tst_inp', 'Y_tr', 'Y_tst']
X_tr = np.load(directorySaving+all_datas[0]+'.npy')
X_test = np.load(directorySaving+all_datas[1]+'.npy')
X_inp_tr = np.load(directorySaving+all_datas[2]+'.npy')
X_inp_test = np.load(directorySaving+all_datas[3]+'.npy')
Y_tr = np.load(directorySaving+all_datas[4]+'.npy')
Y_test = np.load(directorySaving+all_datas[5]+'.npy')
print('done loading data i.e. the train-test split!')
print(X_tr.shape)
print(X_test.shape)
print(X_inp_tr.shape)
print(X_inp_test.shape)


# for i,img in enumerate(X_inp_test):
# 	misc.imsave('./forFindingBases/%d.jpeg'%i,img)

#some intializations before we actually make the poisons
allPoisons = []
alldiffs = []
directoryForPoisons = '/home/tdteach/data/dogfish/poisonImages/'
if not os.path.exists(directoryForPoisons):
    os.makedirs(directoryForPoisons)

for i in range(len(X_test)):
    diff = 100
    maxTriesForOptimizing = 10
    counter = 0
    targetImg = X_inp_test[i]
    usedClosest = False
    while (diff > threshold) and (counter < maxTriesForOptimizing):
        #if target is fish, the poison base should be a dog
        if not startFromClosest and Y_test[i] == 1 and counter<len(class_dog_base_id_in_test):
            baseImg = X_inp_test[class_dog_base_id_in_test[counter]]
        elif not startFromClosest and Y_test[i] == 0 and counter<len(class_fish_base_id_in_test):
            baseImg = X_inp_test[class_fish_base_id_in_test[counter]]
        else:
            if not usedClosest:
                ind = closest_to_target_from_class( classBase = 1 - Y_test[i] , targetFeatRep= X_test[i] ,allTestFeatReps=X_test, allTestClass=Y_test)
                baseImg = X_inp_test[ind]
                usedClosest = True
            else:
                print('Using random base!')
                classBase = 1 - Y_test[i]
                possible_indices = np.argwhere(Y_test == classBase)[:,0]
                ind = np.random.randint(len(possible_indices))
                ind = possible_indices[ind]
                baseImg = X_inp_test[ind]
        img, diff = do_optimization(targetImg, baseImg, MaxIter=1500,coeffSimInp=0.2, saveInterim=False, imageID=i, objThreshold=2.9)
        print('built poison for target %d with diff: %.5f'%(i,diff))
        counter += 1
        # save the image to file and keep statistics
    allPoisons.append(img)
    alldiffs.append(diff)
    name = "%d_%.5f"%(i,diff)
    misc.imsave(directoryForPoisons+name+'.jpeg', img)

allPoisons = np.array(allPoisons)
alldiffs = np.array(alldiffs)
np.save('all_poisons.npy', allPoisons)
np.save('alldiffs.npy', alldiffs)


