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

if (platform == "linux") or (platform == "linux2"):
    print("Linux Detected")
    multiprocessing.set_start_method("spawn")
elif platform == "darwin":
    sys.exit("OSx not supported")
elif platform == "win32":
    print("Windows Detected")

def apply_pruning(layer):
    if (layer.name[-5:] != "prune"):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

def prep_data():
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

    return x_val,y_val,x_sim,y_sim,x_train,y_train

def create_model():
    #########  DEF MODEL  #############

    shape=(28,28,1)
    In = layers.Input(shape=shape,name="OnlyIP")
    Flt = layers.Flatten()(In)
    x1 = Dense(300,name="dense1_prune")(Flt)
    x2 = Dense(100,name="dense2_prune")(x1)
    #x3 = Dense(100,name="dense3_prune")(x2)
    x = Dense(10,name="OPnodo")(x2)
    model = Model([In],[x], name="WOWOWOWOOW")

    return model

def initModel_With_layers(L1,L2,L3):
    ######### RE-DEF MODEL ############
    shape=(28,28,1)
    In = layers.Input(shape=shape,name="OnlyIP")
    Flt = layers.Flatten()(In)
    x1 = Dense(L1,name="dense1_prune")(Flt)
    x2 = Dense(L2,name="dense2_prune")(x1)
    #x3 = Dense(L3,name="dense3_prune")(x2)
    x = Dense(10,name="OPnodo")(x2)
    model = Model([In],[x])
    return model

########### COMPILE MODEL ###########

def compile_np_model(model):
    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    return model

def train_eval_np_model(model,x_train,y_train,epochs,base_dir):

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(base_dir+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)
    model.fit(x_train,y_train, epochs=epochs,callbacks=[model_checkpoint_callback])
    unpruned_val = model.evaluate(x_val,y_val)

    return model, unpruned_val

######### INITIAL SPASIRTY PRUNE ########

def train_p_model(model,x_train,y_train,sparsity_pruning_epochs,prune_logdir):
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

    return model_for_pruning

def strip_save_model(model_for_pruning,base_dir):

    ### SAVE SPARSLY PRUNED MODEL ###

    model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    tf.keras.models.save_model(model_for_pruning, 
                               base_dir+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    model_for_pruning.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model_for_pruning

def save_param_file(model_for_pruning,base_dir):

    ### SAVE PARAMS TO FILE ####

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model_for_pruning.trainable_weights])
    with open(base_dir+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))

def compare_models(model,model_for_pruning):
    ##### Test and compare final models ######

    sparse_pruned_val = model_for_pruning.evaluate(x_val,y_val)

    prune_loop = 0
    model.load_weights(base_dir+'pruned_n_stripped_model.h5')

    return model,sparse_pruned_val

def find_neurons_2_prune(weights,model_layers,x_sim,max_dev,layer_percent_consideration,conn):

    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    model.set_weights(weights)

    layer_consideration = layer_percent_consideration/100

    ####### Find Neurons to prune ########

    i = 0
    submodels = []
    for layer in model.layers:
        if layer.name[-5:] == "prune":
            submodels.append(Model(inputs=model.input, outputs=model.layers[i].output))
        i+=1

    lowest_activations = []

    for submodel in submodels:
        layer_units = submodel.layers[-1].units
        trim_ammount = int(layer_consideration*layer_units)
        if trim_ammount < 0:
            trim_ammount = 1
        submodel_prediction = np.absolute(submodel.predict(x_sim))
        averaged_array = np.average(submodel_prediction,axis=0)
        lowest_10_values_location = np.argsort(averaged_array)[:trim_ammount]
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

    for i in range(len(to_remove)):
        if len(to_remove[i]) == model_layers[i]:
            to_remove[i] = []

    conn.send([to_remove])
    conn.close()

def remove_neurons_n_eval(weights,model_layers,to_remove,x_val,y_val,conn):
    looping = True
    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    model.set_weights(weights)
    ########### Prune Neurons #################
    i=0
    surgeon = Surgeon(model)
    for layer in model.layers:
        if layer.name[-5:] == "prune":
            if len(to_remove[i]) != 0:
                surgeon.add_job('delete_channels', layer, channels=to_remove[i])
                i+=1
            else:
                looping = False

    new_model = surgeon.operate()

    model = new_model

    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    pruned_val = model.evaluate(x_val,y_val)

    for i in range(len(to_remove)):
        model_layers[i] = model_layers[i] - len(to_remove[i])

    conn.send([model.get_weights(),model_layers,pruned_val,looping])
    conn.close()

def retrain(weights,model_layers,x_train,y_train,retrain_epochs,prune_logdir,prune_loop,x_val,y_val,conn):
    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    model.set_weights(weights)
    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    ########### RETRAIN #################

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+str(prune_loop)+'\\'+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)
    model.fit(x_train,y_train, epochs=retrain_epochs,callbacks=[model_checkpoint_callback])
    retrained_val = model.evaluate(x_val,y_val)

    conn.send([model.get_weights(),model_layers,retrained_val])
    conn.close()

def compare_n_store(unpruned_val,sparse_pruned_val,pruned_val,retrained_val,prune_logdir,prune_loop,conn):
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

    conn.send([unpruned_val])
    conn.close()

def sparse_retrain(weights,model_layers,x_train,y_train,sparsity_pruning_retrain_epochs,prune_logdir,prune_loop,conn):
    
    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    model.set_weights(weights)
    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

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
    
    model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    tf.keras.models.save_model(model_for_pruning, 
                               prune_logdir+str(prune_loop)+'\\'+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    model_for_pruning.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    conn.send([model_for_pruning.get_weights(),model_layers])
    conn.close()

def save_curr_model(weights,model_layers,prune_logdir,prune_loop,conn):
    
    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    model.set_weights(weights)
    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    tf.keras.models.save_model(model, 
                               prune_logdir+str(prune_loop)+'\\'+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    conn.send([0])
    conn.close()

def save_num_of_params(weights,model_layers,prune_loop,prune_logdir,conn):
    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    model.set_weights(weights)
    ##### SAVE PARAMS #####
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    with open(prune_logdir+str(prune_loop)+'\\'+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))
        
    conn.send([0])
    conn.close()

    


if __name__ == "__main__":
    ########## INIT PARAMS #########
    
    model_layers = [300,100,0]

    epochs=10
    sparsity_pruning_epochs = 10
    retrain_epochs = 5
    sparsity_pruning_retrain_epochs = 5
    pruning_loops = 20
    max_dev = 0.05  ### IMPORTANT !!!! max deviation from the lowest activation in a given layer
    layer_percent_consideration = 10 # percentage of neuron layer considered for pruning
    
    
    accu_drop = 0.01 ### IMPORTANT !!!! accuracy drop allowed before retrain and stopping point

    base_dir = 'C:\\temp\\mnist\\LE-NET-300\\'
    prune_logdir = 'C:\\temp\\mnist\\LE-NET-300\\pruned\\'
    finished_dir = 'C:\\temp\\mnist\\LE-NET-300\\finished\\'

    ## data setup
    x_val,y_val,x_sim,y_sim,x_train,y_train = prep_data()

    ## model setup
    model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
    ## model train
    model = compile_np_model(model)

    model, unpruned_val = train_eval_np_model(model,x_train,y_train,epochs,base_dir)

    init_val = copy.deepcopy(unpruned_val)

    ## model prune
    model_for_pruning = train_p_model(model,x_train,y_train,sparsity_pruning_epochs,prune_logdir)
    model_for_pruning = strip_save_model(model_for_pruning,base_dir)

    ## save model
    save_param_file(model_for_pruning,base_dir)
    model,sparse_pruned_val = compare_models(model,model_for_pruning)

    weights = model.get_weights()

    ###pruning loop
    prune_loop = 0
    looping = True
    while looping:#prune_loop < pruning_loops:

        tf.keras.backend.clear_session
        try:
            os.mkdir(prune_logdir+str(prune_loop))
        except:
            pass

        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=find_neurons_2_prune, args=(weights,model_layers,x_sim,max_dev,layer_percent_consideration,child_conn))
        
        reader_process.start()
        
        remove_points_returned = parent_conn.recv()
        
        remove_points = remove_points_returned[0]
        
        reader_process.join()
        
        #to_remove = find_neurons_2_prune(model)
        
        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=remove_neurons_n_eval, args=(weights,model_layers,remove_points,x_val,y_val,child_conn))
        
        reader_process.start()
        
        returned_vars = parent_conn.recv()
        
        weights,model_layers,pruned_val,looping = returned_vars
        
        reader_process.join()
        
        #model, pruned_val = remove_neurons_n_eval(model,to_remove,x_val,y_val) retrain(weights,model_layers,retrain_epochs,prune_logdir,prune_loop,x_val,y_val,conn)

        if unpruned_val[1] > pruned_val[1] +  accu_drop:

            parent_conn, child_conn = multiprocessing.Pipe()
        
            reader_process  = multiprocessing.Process(target=retrain, args=(weights,model_layers,x_train,y_train,retrain_epochs,prune_logdir,prune_loop,x_val,y_val,child_conn))
        
            reader_process.start()
        
            returned_vars = parent_conn.recv()
        
            weights,model_layers,retrained_val = returned_vars
        
            reader_process.join()

            with open(prune_logdir+str(prune_loop)+'RT', 'w') as f:
                f.write(str('prune_occured'))

        else:
            retrained_val = unpruned_val
        
        #model,retrained_val = retrain(weights,model_layers,retrain_epochs,prune_logdir,prune_loop,x_val,y_val,conn)
        
        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=compare_n_store, args=(unpruned_val,sparse_pruned_val,pruned_val,retrained_val,prune_logdir,prune_loop,child_conn))
        
        reader_process.start()
        
        returned_vars = parent_conn.recv()
        
        unpruned_val_w = returned_vars[0]
        
        reader_process.join()
        
        #unpruned_val = compare_n_store(unpruned_val,sparse_pruned_val,pruned_val,retrained_val,prune_logdir,prune_loop)
        
        if unpruned_val[1] > pruned_val[1] +  0.01:

            parent_conn, child_conn = multiprocessing.Pipe()
        
            reader_process  = multiprocessing.Process(target=sparse_retrain, args=(weights,model_layers,x_train,y_train,sparsity_pruning_retrain_epochs,prune_logdir,prune_loop,child_conn))
        
            reader_process.start()
        
            returned_vars = parent_conn.recv()
        
            weights,model_layers = returned_vars
        
            reader_process.join()

        else:

            parent_conn, child_conn = multiprocessing.Pipe()
        
            reader_process  = multiprocessing.Process(target=save_curr_model, args=(weights,model_layers,prune_logdir,prune_loop,child_conn))
        
            reader_process.start()
        
            returned_vars = parent_conn.recv()
        
            not_used = returned_vars
        
            reader_process.join()
        
        #model_for_pruning = sparse_retrain(model,x_train,y_train,sparsity_pruning_retrain_epochs,prune_logdir,prune_loop)
        
        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=save_num_of_params, args=(weights,model_layers,prune_loop,prune_logdir,child_conn))
        
        reader_process.start()
        
        returned_vars = parent_conn.recv()
        
        notused = returned_vars
        
        reader_process.join()
        
        #save_num_of_params(model_for_pruning,prune_loop)
        
        # end of loop
        
        with open(prune_logdir+str(prune_loop)+'\\'+'model_layers.txt', 'w') as f:
            f.write(str(model_layers[0])+'\n')
            f.write(str(model_layers[1])+'\n')
            f.write(str(model_layers[2])+'\n')

        unpruned_val = unpruned_val_w

        model = initModel_With_layers(model_layers[0],model_layers[1],model_layers[2])
        
        model.load_weights(prune_logdir+str(prune_loop)+'\\'+'pruned_n_stripped_model.h5')
        
        prune_loop += 1

        a,b = init_val[1] - accu_drop, retrained_val[1]

        if init_val[1] - accu_drop > retrained_val[1]:
            print("Accuracy Dropped Too Far")
            break

    #final test
    #make dir
    os.mkdir(finished_dir)
    #save model
    tf.keras.models.save_model(model, finished_dir+'\\'+'finished_model.h5', include_optimizer=False)
    ##### SAVE PARAMS #####
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    with open(finished_dir+'\\'+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))
    #save layers
    with open(finished_dir+'\\'+'model_layers.txt', 'w') as f:
            f.write(str(model_layers[0])+'\n')
            f.write(str(model_layers[1])+'\n')
            f.write(str(model_layers[2])+'\n')
    #save eval
    model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    finished = model.evaluate(x_val,y_val)
    init_val

    with open(finished_dir+'\\'+'EvalResults.txt', 'w') as f:
        f.write(str("Initial Model:"))
        f.write(str(init_val))
        f.write(str("\nFinished model:"))
        f.write(str(finished))