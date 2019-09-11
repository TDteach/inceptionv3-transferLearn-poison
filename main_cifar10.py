import util_one_shot_kill_attack as util
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy import misc
from os import listdir
from tensorflow.python.platform import gfile
from scipy.spatial.distance import cdist
from datetime import datetime
import os



def decode_img(img_array):
    img = np.reshape(img_array,[3,32,32])
    img = np.transpose(img,[1,2,0])
    img = np.expand_dims(img,axis=0)
    return img

def load_images_from_dataset(selected_lbs, data_dir):
    def unpickle(file):
        import pickle
        with open(file,'rb') as fo:
            ret = pickle.load(fo,encoding='bytes')
        return ret

    ret_tr_x = []
    ret_te_x = []
    ret_tr_y = []
    ret_te_y = []

    import os
    filenames = [
            os.path.join(data_dir,'data_batch_%d' % i)
            for i in range(1,6)
            ]
    for d in filenames:
        data_dict = unpickle(d)
        data = data_dict[b'data']
        labels = data_dict[b'labels']

        for lb, im in zip(labels,data):
            if lb in selected_lbs:
                ret_tr_x.append(decode_img(im))
                ret_tr_y.append(lb)

    f_path = os.path.join(data_dir,'test_batch')
    data_dict = unpickle(f_path)
    data = data_dict[b'data']
    labels = data_dict[b'labels']

    for lb, im in zip(labels,data):
        if lb in selected_lbs:
            ret_te_x.append(decode_img(im))
            ret_te_y.append(lb)

    return ret_tr_x, ret_tr_y, ret_te_x, ret_te_y


def get_feat_reps(X, class_t):
    feat_tensor_name = 'v0/cg/affine1/xw_plus_b:0'
    input_tensor_name = 'input_image:0'

    sess = tf.Session()
    util.create_graph(graphDir)

    feat_tensor = sess.graph.get_tensor_by_name(feat_tensor_name)
    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

    res = []
    for i,x in enumerate(X):
        res.append(sess.run(feat_tensor, feed_dict={input_tensor:x}))
        if i % 50 == 0:
            print('finished %d\'th example of %s' % (i, class_t))
    res = np.array(res)

    tf.reset_default_graph()
    sess.close()
    return res

def do_optimization(targetImg, baseImg, MaxIter=200,coeffSimInp=0.25, saveInterim=False, imageID=0, objThreshold = 2.9):
    """
    Returns the poison image and the difference between the poison and target in feature space.
    Parameters
    ----------
    targetImg : ndarray
        the input image of the target from the  test set.
    baseImg : ndarray
        the input image of the base class (this should have a differet class than the target)
    MaxIter : integer
        this is the maximum number of fwd backward iterations
    coeffSimInp : flaot
        the coefficient of similarity to the base image in input image space relative to the
        similarity to the feature representation of the target when everything is normalized
        the objective function of the optimization is:
                || f(x)-f(t) ||^2 + coeffSimInp * || x-b ||^2
    objThreshold: float
        the threshold for the objective functoin, when the obj func falls below this, the
        optimization is stopped even if the MaxIter is not met.
    Returns
    -------
    old_image, finalDiff : ndarray, float
        The poison in uin8 format
        The difference in feature space measure by the 2-norm
    """

    coeffSimInp = 0.1
    MaxIter = 3000

    #parameters for cifar10:
    Adam = False
    decayCoef = 0.5                 #decay coeffiencet of learning rate
    learning_rate = 500.0*255      #iniital learning rate for optimiz
    stopping_tol = 1e-10            #for the relative change
    EveryThisNThen = 20             #for printing reports
    M = 40                          #used for getting the average of last M objective function values
    BOTTLENECK_TENSOR_NAME = 'v0/cg/affine1/xw_plus_b'
    BOTTLENECK_TENSOR_SIZE = 192
    INPUT_TENSOR_NAME = 'input_image:0'

    targetImg = targetImg.astype(np.float32)
    baseImg = baseImg.astype(np.float32)
    baseImg = 0.7*baseImg+0.3*targetImg

    #calculations for getting a reasonable value for coefficient of similarity of the input to the base image
    bI_shape = np.squeeze(baseImg).shape
    coeff_sim_inp = coeffSimInp*(BOTTLENECK_TENSOR_SIZE/float(bI_shape[0]*bI_shape[1]*bI_shape[2]))**2
    print('coeff_sim_inp is:', coeff_sim_inp)

    #load the inception v3 graph
    sess = tf.Session()
    graph = util.create_graph(graphDir)

    #add some of the needed operations
    featRepTensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME+':0')
    inputImgTensor = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
    tarFeatRepPL = tf.placeholder(tf.float32,[1,BOTTLENECK_TENSOR_SIZE])
    forward_loss = tf.norm(featRepTensor - tarFeatRepPL)
    grad_op = tf.gradients(forward_loss, inputImgTensor)

    #initializations
    last_M_objs = []
    rel_change_val = 1e5
    targetFeatRep = sess.run(featRepTensor, feed_dict={inputImgTensor: targetImg})      #get the feature reprsentation of the target
    old_image = baseImg                                                                 #set the poison's starting point to be the base image
    old_featRep = sess.run(featRepTensor, feed_dict={inputImgTensor: baseImg})      #get the feature representation of current poison
    old_obj = np.linalg.norm(old_featRep - targetFeatRep) + coeff_sim_inp*np.linalg.norm(old_image - baseImg)
    last_M_objs.append(old_obj)

    #intializations for ADAM
    if Adam:
        m = 0.
        v = 0.
        t = 0

    #optimization being done here
    for iter in range(MaxIter):
        #save images every now and then
        if iter % EveryThisNThen == 0:
            the_diffHere = np.linalg.norm(old_featRep - targetFeatRep)      #get the diff
            theNPimg = old_image                                            #get the image
            print("iter: %d | diff: %.3f | obj: %.3f"%(iter,the_diffHere,old_obj))
            print(" (%d) Rel change =  %0.5e   |   lr = %0.5e |   obj = %0.10e"%(iter,rel_change_val,learning_rate,old_obj))
            if saveInterim:
                name = '%d_%d_%.5f.jpeg'%(imageID,iter,the_diffHere)
                misc.imsave('./interimPoison/'+name, np.squeeze(old_image).astype(np.uint8))
            # plt.imshow(np.squeeze(old_image).astype(np.uint8))
            # plt.show()

        # forward update gradient update
        if Adam:
            new_image,m,v,t = util.adam_one_step(sess=sess,grad_op=grad_op,m=m,v=v,t=t,currentImage=old_image,featRepTarget=targetFeatRep,tarFeatRepPL=tarFeatRepPL,inputCastImgTensor=inputImgTensor,learning_rate=learning_rate)
        else:
            new_image = util.do_forward(sess=sess,grad_op=grad_op,inputCastImgTensor=inputImgTensor, currentImage=old_image,featRepCurrentImage=old_featRep,featRepTarget=targetFeatRep,tarFeatRepPL=tarFeatRepPL,learning_rate=learning_rate)

        #print(np.max(new_image))
        #print(np.min(new_iage))
        #print(np.linalg.norm(new_image-old_image))

        # The backward step in the forward-backward iteration
        new_image = util.do_backward(baseInpImage=baseImg,currentImage=new_image,coeff_sim_inp=coeff_sim_inp,learning_rate=learning_rate,eps=0.1)

        # check stopping condition:  compute relative change in image between iterations
        rel_change_val =  np.linalg.norm(new_image-old_image)/np.linalg.norm(new_image)
        if (rel_change_val<stopping_tol) or (old_obj<=objThreshold):
            break

        # compute new objective value
        new_featRep = sess.run(featRepTensor, feed_dict={inputImgTensor: new_image})
        new_obj = np.linalg.norm(new_featRep - targetFeatRep) + coeff_sim_inp*np.linalg.norm(new_image - baseImg)
        #new_obj = np.linalg.norm(new_featRep - targetFeatRep)

        if Adam:
            learning_rate = 0.1*255.
            old_image = new_image
            old_obj = new_obj
            old_featRep = new_featRep
        else:

            avg_of_last_M = sum(last_M_objs)/float(min(M,iter+1)) #find the mean of the last M iterations
            # If the objective went up, then learning rate is too big.  Chop it, and throw out the latest iteration
            if  new_obj >= avg_of_last_M and (iter % (M//2) == 0):
                learning_rate *= decayCoef
                new_image = old_image
            else:
                old_image = new_image
                old_obj = new_obj
                old_featRep = new_featRep

            if iter < M-1:
                last_M_objs.append(new_obj)
            else:
                #first remove the oldest obj then append the new obj
                del last_M_objs[0]
                last_M_objs.append(new_obj)
            if iter > MaxIter:
                m = 0.
                v = 0.
                t = 0
                Adam = True

    finalDiff = np.linalg.norm(old_featRep - targetFeatRep)
    print('final diff: %.3f | final obj: %.3f'%(finalDiff,old_obj))
    #close the session and reset the graph to clear memory
    sess.close()
    tf.reset_default_graph()

    return np.squeeze(old_image).astype(np.uint8), finalDiff


"""
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
"""

graphDir = './alexnetModel/cifar10_alexnet.pb'
dataDir = '/home/tdteach/data/airplanfrog/'
directorySaving = dataDir+'XY/'
firstTime = False
threshold = 3.5 #threshold for L2 distance in feature space

if firstTime:
    if not os.path.exists(directorySaving):
        os.makedirs(directorySaving)
    X_inp_tr, Y_tr, X_inp_test, Y_test = load_images_from_dataset([0,6],'/home/tdteach/data/CIFAR-10')
    X_tr = get_feat_reps(X_inp_tr, 'train')
    X_test = get_feat_reps(X_inp_test, 'test')
    np.save(directorySaving+'X_tr_feats.npy', X_tr)
    np.save(directorySaving+'X_tst_feats.npy', X_test)
    np.save(directorySaving+'X_tr_inp.npy', X_inp_tr)
    np.save(directorySaving+'X_tst_inp.npy', X_inp_test)
    np.save(directorySaving+'Y_tr.npy', Y_tr)
    np.save(directorySaving+'Y_tst.npy', Y_test)


all_datas = ['X_tr_feats', 'X_tst_feats', 'X_tr_inp', 'X_tst_inp', 'Y_tr', 'Y_tst']
X_tr = np.load(directorySaving+all_datas[0]+'.npy')
X_test = np.load(directorySaving+all_datas[1]+'.npy')
X_inp_tr = np.load(directorySaving+all_datas[2]+'.npy')
X_inp_test = np.load(directorySaving+all_datas[3]+'.npy')
Y_tr = np.load(directorySaving+all_datas[4]+'.npy')
Y_test = np.load(directorySaving+all_datas[5]+'.npy')
print('done loading data i.e. the train-test split!')

allPoisons = []
alldiffs = []
directoryForPoisons = '/home/tdteach/data/airplanfrog/poisonImages/'
if not os.path.exists(directoryForPoisons):
    os.makedirs(directoryForPoisons)

for i in range(9, min(len(X_test),10)):
    diff = 100
    maxTriesForOptimizing = 10
    counter = 0
    targetImg = X_inp_test[i]
    usedClosest = False
    if Y_test[i] == 0:
        classBase = 6
    elif Y_test[i] == 6:
        classBase = 0
    while (diff > threshold) and (counter < maxTriesForOptimizing):
        if not usedClosest:
            ind = util.closest_to_target_from_class( classBase = classBase, targetFeatRep= X_test[i] ,allTestFeatReps=X_test, allTestClass=Y_test)
            baseImg = X_inp_test[ind]
            usedClosest = True
        else:
            print('Using random base!')
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



