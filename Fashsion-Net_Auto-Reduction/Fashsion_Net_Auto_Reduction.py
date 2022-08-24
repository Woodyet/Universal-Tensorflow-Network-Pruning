
# TensorFlow and tf.keras
import tensorflow as tf
from tqdm import tqdm
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import json
import copy
import pandas as pd
import tensorflow_model_optimization as tfmot
import skimage.measure
import tempfile
from collections import Counter
import time
import random
import string
import os
import statistics
convComps = 4


def apply_weights_conv2D(W2RLayer,W2RPrunedAll,pruned_keras_file,through,allremovedfilters):
    '''
    (layer,neuron,most recent version of the model)
    '''
    W2RPruned = W2RPrunedAll[through]
    
    model = tf.keras.models.load_model(pruned_keras_file) 
    time.sleep(1)
    postweights = model.get_weights()
    
    preweights = copy.deepcopy(postweights)
    
    
    #remove channel 0 from OP.
    chanremove = W2RPruned #channel to remove from
    LTRF = W2RLayer*2 #layer to remove from 
    
    #REMOVING WEIGHTS
    
    alldems = []
    for i in range(3):
        oki8 =  copy.deepcopy(np.delete(postweights[LTRF][i][0],chanremove,1))
        oki9 =  copy.deepcopy(np.delete(postweights[LTRF][i][1],chanremove,1))
        oki10 = copy.deepcopy(np.delete(postweights[LTRF][i][2],chanremove,1))
    
        alldems.append(np.append([oki8, oki9], [oki10], axis=0))
    
    postweights[LTRF] = np.append([alldems[0], alldems[1]], [alldems[2]], axis=0)
    
    #REMOVING B[LTRF]IAS
    
    postweights[LTRF+1] = np.delete(postweights[LTRF+1],chanremove)
    
    #remove a channel from IP.
    
    postweights[LTRF+2] = np.delete(postweights[LTRF+2],chanremove,0)
     
    #define model
    
    modelfrank = reducemodel(model,W2RPruned,False,through,W2RPrunedAll,allremovedfilters)
    
    
    modelfrank.set_weights(postweights)
    
    modelfrank = compare_models(model,modelfrank)
    
    return modelfrank

def reducemodel(oldmodel,change_by,p2,through,removed,allremovedfilters):
    
    model = redefinemodelautobuild(oldmodel,len(change_by),p2,through,removed,allremovedfilters)
    
    return model

def redefinemodelautobuild(model,reduction,p2,i,channelsremoved2,channelsremoved):
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28,1)),
    tf.keras.layers.Conv2D(convComps-reduction, (3, 3), activation='relu',name="pruneable1"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])
    return model

def apply_pruning(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

def combine_removed_nodes(base,new,layersize):
    Original =  list(range(layersize))
    NewMap =    list(range(layersize))
    
    for i in base[::-1]:
        Original.pop(i)
    
    for i in new[::-1]:
        Original.pop(i)
    
    for i in Original[::-1]:
        NewMap.pop(i)
    
    return NewMap

def find_max(pred):
    max = pred.max()
    min = pred.min()
    if min * -1 > max:
        forscale = min * -1
    else:
        forscale = max
    midpoint = max + (min-max)/2
    return forscale,midpoint

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def compare_models(a,b):
    
    a.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    b.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return b

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0

train_images = train_images.reshape((train_images.shape[0], 28, 28,1))
test_images = test_images.reshape((test_images.shape[0], 28, 28,1))

for skilly in range(32):
    convComps = skilly+1
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28,1)),
        tf.keras.layers.Conv2D(convComps, (3, 3), activation='relu',name="pruneable1"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
        ])


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    #end_step = np.ceil(60000 / 32).astype(np.int32) * 5
    #
    #pruning_params = {
    #  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
    #                                                           final_sparsity=0.80,
    #                                                           begin_step=0,
    #                                                           end_step=end_step)
    #  }
    #
    #
    #model_for_pruning = tf.keras.models.clone_model(
    #    model,
    #    clone_function=apply_pruning,
    #)
    #
    #model_for_pruning.compile(optimizer='adam',
    #              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #              metrics=['accuracy'])
    #
    #model_for_pruning.fit(train_images, 
    #                      train_labels,
    #                      batch_size=32, 
    #                      epochs=5,
    #                      callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
    #
    #model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    #_, pruned_keras_file = tempfile.mkstemp('.h5')
    #tf.keras.models.save_model(model_for_pruning, pruned_keras_file, include_optimizer=False)
    #print('Saved pruned Keras model to:', pruned_keras_file)
    #
    #model = tf.keras.models.load_model(pruned_keras_file) 

    allremovedfilters = []
    #sort
    n = 0
    for i in range(len(model.layers)):
        allremovedfilters.append([])

    submodels = []
    for i in range(len(model.layers)):
        if model.layers[i].name[:len("pruneable")] == "pruneable":
            layer_type = json.loads(model.layers[i]._tracking_metadata)["class_name"]
            submodels.append(Model(inputs=model.input, outputs=model.layers[i].output))
        else:
            submodels.append([])


    predictions = []
    LayerPred = []
    ShannonEntrop = []

    for submodel in tqdm(submodels):
        if submodel != []:
            base = submodel.predict(test_images[0:1000])
            predictions.append(base)
            ShannonEntrop.append([])
            LayerPred.append([])
        else:
            predictions.append([])
            ShannonEntrop.append([])
            LayerPred.append([])

    lp = 0

    for layerprediction in tqdm(predictions):
        F = True
        for specpred in layerprediction:
            if F == True:
                LayerPred[lp] = copy.deepcopy(specpred)
                F = False
            else:
                LayerPred[lp] += copy.deepcopy(specpred)
            shannons = []
            if len(specpred.shape) > 1:
                for i in range(specpred.shape[2]):
                    shannons.append(skimage.measure.shannon_entropy(specpred[:,:,i]))
            else:
                pass
            ShannonEntrop[lp].append(shannons)

        lp +=1
    all_shans_summed = []
    for shan in ShannonEntrop:
        shans_summed = []
        F = True
        if len(shan) > 1:
            for els in shan:
                if F==True:
                    shans_summed = els
                    F=False
                else:
                    shans_summed = [sum(x) for x in zip(*[shans_summed,els])]
                    pass
        temp = []
        for i in shans_summed:
            temp.append(i/len(shan))
        all_shans_summed.append(temp)
   



    lp = 0
    for pred in LayerPred:
        if pred != []:
            LayerPred[lp] = pred/predictions[0].shape[0]
        lp +=1

    layer = 0
    print("making pretty pictures")
    all_raw = []
    for pred in tqdm(LayerPred):
        if pred == []:
            pass
        elif len(pred.shape) > 1:
            forscale,midpoint = find_max(pred)
            for chanloc in range(pred.shape[2]):
                #toplot = np.pad(pred[:,:,chanloc], 1, pad_with, padder=0)
                #toplot[0,0] = forscale
                #toplot[0,pred.shape[1]+1] = forscale * -1

                #plt.imshow(toplot)
                #plt.show()
                raw = np.array(pred[:,:,chanloc])
                all_raw.append(pred[:,:,chanloc])
                #plt.savefig(Prefix + 'L' +str(layer) + 'C' + str(chanloc) + '.png')
                #plt.clf()
        else :
            #forscale,midpoint = find_max(pred)
            #oki = np.zeros(pred.size)
            #oki.fill(0)
            #toplot = np.append([oki,pred],[oki], axis=0)
            #toplot[0] = forscale
            #toplot[-1] = forscale * -1
            pass
            #plt.imshow(toplot)
            #plt.show()
            #plt.savefig(Prefix + 'L' +str(layer) + 'FC' + '.png')
            #plt.clf()


        layer +=1

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    j=0
    letters = string.ascii_lowercase
    foldername = ''.join(random.choice(letters) for i in range(10))
    os.mkdir("C:\\temp\\shannons\\"+str(skilly))
    f = open("C:\\temp\\shannons\\"+str(skilly)+"\\accuracy.txt", "a")
    f.write('\nTest accuracy:'+str(test_acc))
    f.close()
    tmp = 0
    oki2 = 0
    compare_against_5 = []
    for i in all_shans_summed[0]:
        tmp+=i
        oki2+=1
        compare_against_5.append(i-5)
    tmp = tmp/oki2
    f = open("C:\\temp\\shannons\\"+str(skilly)+"\\AverageSHanperChan.txt", "a")
    f.write('Average:'+str(tmp))
    f.close()

    f = open("C:\\temp\\shannons\\"+str(skilly)+"\\MediansSHanperChan.txt", "a")
    f.write('median:'+str(statistics.median(all_shans_summed[0])))
    f.close()

    comp = 0
    for i in compare_against_5:
        comp+=i

    f = open("C:\\temp\\shannons\\"+str(skilly)+"\\Compare_aginst_5.txt", "a")
    f.write(str(compare_against_5))
    f.write(str(comp))
    f.close()

    for i in all_shans_summed[0]:
        print(i)
        plt.imshow(all_raw[j])
        shan = str(float("{0:.6f}".format(i)))
        #Letter = chr(ord("A")+j)
        plt.savefig("C:\\temp\\shannons\\"+str(skilly)+"\\"+shan.replace(".", "-")+"_"+str(j)+'.png')
        plt.clf()
        j+=1

averagedShanns = 0 
for pred in all_shans_summed[0]:
    averagedShanns+=pred

averagedShanns = averagedShanns/len(all_shans_summed[0])

remove_points = []
for pred in all_shans_summed:
    nodes_to_remove = []
    if pred == []:
        pass
    elif len(pred) > 1:     
        nodes_to_remove.append(int(input("which to remove?")))
        oki =5
    else:
        j = 0
        for element in pred:
            if element < prune_threshold:
                nodes_to_remove.append(j)
            j+=1

    nodes_to_remove.sort()
    remove_points.append(nodes_to_remove)




for i in tqdm(range(len(model.layers))):
    
    test = model.layers[i].get_weights()     
    
    if json.loads(model.layers[i]._tracking_metadata)["class_name"] == "Conv2D":
        if model.layers[i].filters - len(remove_points[i]) < 1:
            remove_points[i].pop()
    
retrain = True
for item in remove_points:
    if len(item) > 0:
        retrain = False

weightloc = 0
for i in tqdm(range(len(model.layers))):

    test = model.layers[i].get_weights()     
    
    if json.loads(model.layers[i]._tracking_metadata)["class_name"] == "Conv2D":
    
        model = apply_weights_conv2D(weightloc,remove_points,pruned_keras_file,i,allremovedfilters)
    
        allremovedfilters[i] = combine_removed_nodes(allremovedfilters[i],remove_points[i],model.layers[i].filters+len(remove_points[i])+len(allremovedfilters[i]))
    
        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model, pruned_keras_file, include_optimizer=False)
        #print('Saved actual pruned Keras model to:', pruned_keras_file)
    if test != []:
        weightloc +=1


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


predictions = probability_model.predict(test_images)

np.argmax(predictions[0])

test_labels[0]



i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()