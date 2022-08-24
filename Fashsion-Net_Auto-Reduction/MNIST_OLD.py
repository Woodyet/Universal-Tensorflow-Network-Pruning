
import tensorflow as tf
print(tf. __version__)
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses
import copy
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
import time
from tensorflow.keras.models import Model
import tempfile
from kerassurgeon.operations import delete_channels
import random
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
from kerassurgeon import Surgeon

import multiprocessing
from sys import platform

def apply_pruning(layer):
    if (layer.name[-4:] != "nodo"):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


#################### PREP DATA ##############################

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()    #get data

x_train = tf.keras.utils.normalize(tf.expand_dims(x_train, axis=3, name=None))
x_test = tf.keras.utils.normalize(tf.expand_dims(x_test, axis=3, name=None))
print(x_train.shape)

x_val = x_train[-2000:,:,:,:] #split data
y_val = y_train[-2000:] 
x_sim = x_train[-8000:-2000,:,:,:] 
y_sim = y_train[-8000:-2000]
x_train = x_train[:-8000,:,:,:] 
y_train = y_train[:-8000]

#########  DEF MODEL  #############

shape=(28,28,1)
In = layers.Input(shape=shape,name="OnlyIP")
Flt = layers.Flatten()(In)
x1 = Dense(100,name="dense1_prune")(Flt)
x2 = Dense(100,name="dense2_prune")(x1)
x3 = Dense(100,name="dense3_prune")(x2)
x = Dense(10,name="OPnodo")(x3)
model = Model([In],[x], name="WOWOWOWOOW")

######### RE-DEF MODEL ############

def initModel_With_layers(L1,L2,L3):
    In = layers.Input(shape=shape,name="OnlyIP")
    Flt = layers.Flatten()(In)
    x1 = Dense(L1,name="dense1")(Flt)
    x2 = Dense(L2,name="dense2")(x1)
    x3 = Dense(L3,name="dense3")(x1)
    x = Dense(10,name="OPnodo")(x3)
    return Model([In],[x])

########### COMPILE MODEL ###########


model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

########## INITIAL TRAINING #########

test_points=1
epochs=10
sparsity_pruning_epochs = 10
retrain_epochs = 3
sparsity_pruning_retrain_epochs = 3
pruning_loops = 10
max_dev = 0.02   ### IMPORTANT !!!! max deviation from the lowest activation in a given layer
base_dir = 'C:\\temp\\mnist\\LE-NET-300\\'
prune_logdir = 'C:\\temp\\mnist\\LE-NET-300\\pruned\\'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(base_dir+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)

model.fit(x_train,y_train, epochs=epochs,callbacks=[model_checkpoint_callback])
unpruned_val = model.evaluate(x_val,y_val)
######### INITIAL SPASIRTY PRUNE ########

end_step = x_train.shape[0]*sparsity_pruning_epochs
pruning_params = {
  'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                           final_sparsity=0.80,
                                                           begin_step=0,
                                                           end_step=end_step)
  }


model_for_pruning = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning,
)

model_for_pruning.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+'model-prune{epoch:02d}.h5',
                                                               save_weights_only=False,save_freq='epoch',
                                                               save_best_only=False,
                                                               verbose=1)
model_for_pruning.fit(x_train,
                      y_train,
                      epochs=sparsity_pruning_epochs,
                      callbacks=[model_checkpoint_callback,tfmot.sparsity.keras.UpdatePruningStep(),tfmot.sparsity.keras.PruningSummaries(log_dir=prune_logdir),])

### SAVE SPARSLY PRUNED MODEL ###

model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

tf.keras.models.save_model(model_for_pruning, 
                           base_dir+'pruned_n_stripped_model.h5', 
                           include_optimizer=False)

### SAVE PARAMS TO FILE ####

trainableParams = np.sum([np.prod(v.get_shape()) for v in model_for_pruning.trainable_weights])
with open(base_dir+'model_params.txt', 'w') as f:
    f.write(str(trainableParams))

model_for_pruning.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

##### Test and compare final models ######

unpruned_val
sparse_pruned_val = model_for_pruning.evaluate(x_val,y_val)

prune_loop = 0
model.load_weights(base_dir+'pruned_n_stripped_model.h5')

##### START PUNING LOOP ######
while True:#prune_loop < pruning_loops:
    try:
        os.mkdir(prune_logdir+str(prune_loop))
    except:
        pass
    ####### Find Neurons to prune ########

    i = 0
    submodels = []
    for layer in model.layers:
        if layer.name[-5:] == "prune":
            submodels.append(Model(inputs=model.input, outputs=model.layers[i].output))
        i+=1

    lowest_activations = []

    for submodel in submodels:
        submodel_prediction = np.absolute(submodel.predict(x_sim))
        averaged_array = np.average(submodel_prediction,axis=0)
        lowest_10_values_location = np.argsort(averaged_array)[:10]
        lowest_10_values = []
        for i in lowest_10_values_location:
            #print(averaged_array[i])
            lowest_10_values.append(averaged_array[i])
        lowest_activations.append([lowest_10_values_location,lowest_10_values])

    to_remove = []

    for lowest_layer in lowest_activations:
        locs,acts = lowest_layer
        retain = 0
        for val in acts:
            if val > acts[0] + max_dev:
                break
            else:
                retain+=1
        rem_locs = locs[:retain]
        to_remove.append(rem_locs)

    ########### Prune Neurons #################

    i=0
    surgeon = Surgeon(model)
    for layer in model.layers:
        if layer.name[-5:] == "prune":
            surgeon.add_job('delete_channels', layer, channels=to_remove[i])
            i+=1

    new_model = surgeon.operate()

    model = new_model

    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    pruned_val = model.evaluate(x_val,y_val)

    ########### RETRAIN #################

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+str(prune_loop)+'\\'+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)
    model.fit(x_train,y_train, epochs=retrain_epochs,callbacks=[model_checkpoint_callback])
    retrained_val = model.evaluate(x_val,y_val)

    ######## Compares #########

    print("Unpruned:")
    print(unpruned_val)
    print("Sparse pruned:")
    print(sparse_pruned_val)
    print("Actually Pruned:")
    print(pruned_val)
    print("retrained:")
    print(retrained_val)

    with open(prune_logdir+str(prune_loop)+'\\'+'EvalResults.txt', 'w') as f:
        f.write(str("Unpruned:"))
        f.write(str(unpruned_val))
        f.write(str("\nSparse pruned:"))
        f.write(str(sparse_pruned_val))
        f.write(str("\nActually Pruned:"))
        f.write(str(pruned_val))
        f.write(str("\nretrained:"))
        f.write(str(retrained_val))

    unpruned_val = retrained_val

    ########### SPARSE_RETRAIN ##############

    end_step = x_train.shape[0]*sparsity_pruning_retrain_epochs
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
      }


    model_for_pruning = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning,
    )

    model_for_pruning.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+str(prune_loop)+'\\'+'model-prune{epoch:02d}.h5',
                                                                   save_weights_only=False,save_freq='epoch',
                                                                   save_best_only=False,
                                                                   verbose=1)
    model_for_pruning.fit(x_train,
                          y_train,
                          epochs=sparsity_pruning_retrain_epochs,
                          callbacks=[model_checkpoint_callback,tfmot.sparsity.keras.UpdatePruningStep(),tfmot.sparsity.keras.PruningSummaries(log_dir=prune_logdir+str(prune_loop)),])

    ### SAVE SPARSLY PRUNED MODEL ###

    model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    tf.keras.models.save_model(model_for_pruning, 
                               prune_logdir+str(prune_loop)+'\\'+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    ##### SAVE PARAMS #####

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model_for_pruning.trainable_weights])
    with open(prune_logdir+str(prune_loop)+'\\'+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))

    model_for_pruning.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    model.load_weights(prune_logdir+str(prune_loop)+'\\'+'pruned_n_stripped_model.h5')

    

    prune_loop += 1