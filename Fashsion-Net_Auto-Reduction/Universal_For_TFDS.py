
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses

from kerassurgeon.operations import delete_channels

import tensorflow_probability as tfp
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

import tensorflow 
import tensorflow as tf
import tensorflow_datasets as tfds
from kerassurgeon import Surgeon

import multiprocessing
from sys import platform
import copy
import time
import tempfile
import os
import numpy as np
import random
import math
from tqdm import tqdm

print(tf. __version__)
# import necessary layers  
from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense 
from tensorflow.keras import Model

from tensorflow.keras import datasets, layers, models

class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, iterable=None, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)

def resize_image(image, shape = (224,224)):
  target_width = shape[0]
  target_height = shape[1]
  initial_width = tf.shape(image)[0]
  initial_height = tf.shape(image)[1]
  im = image
  ratio = 0
  if(initial_width < initial_height):
    ratio = tf.cast(256 / initial_width, tf.float32)
    h = tf.cast(initial_height, tf.float32) * ratio
    im = tf.image.resize(im, (256, h), method="bicubic")
  else:
    ratio = tf.cast(256 / initial_height, tf.float32)
    w = tf.cast(initial_width, tf.float32) * ratio
    im = tf.image.resize(im, (w, 256), method="bicubic")
  width = tf.shape(im)[0]
  height = tf.shape(im)[1]
  startx = width//2 - (target_width//2)
  starty = height//2 - (target_height//2)
  im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
  return im

def resize_with_crop(image, label):
    i = image
    i = tf.cast(i, tf.float32)
    i = resize_image(i, (224,224))
    i = tf.keras.applications.vgg16.preprocess_input(i)
    return (i, label)


def load_imagenet(section):
    # Get imagenet labels
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    # Set data_dir to a read-only storage of .tar files
    # Set write_dir to a w/r storage
    data_dir = '/data/PublicDataSets/ImageNet/ILSVRC2012/'
    write_dir = '/data/scratch/eex869/IMAGENET_2012/'

    # Construct a tf.data.Dataset
    download_config = tfds.download.DownloadConfig(
                          extract_dir=os.path.join(write_dir, 'extracted'),
                          manual_dir=data_dir
                      )
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': download_config,
    }

    ds = tfds.load('imagenet2012', #_subset', 
                   data_dir=os.path.join(write_dir, 'data'),         
                   split=section, 
                   shuffle_files=False, 
                   download=True, 
                   as_supervised=True,
                   download_and_prepare_kwargs=download_and_prepare_kwargs)
               
    ds = ds.map(resize_with_crop)
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)
    return ds

def apply_pruning(layer):
    if (layer.name[-5:] != "prune"):
        if isinstance(layer, tf.keras.layers.Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

def prep_data_cifar10():
    #################### PREP DATA ##############################

    (x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()    #get data

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_sim = x_test[-2000:,:,:,:]
    y_sim = y_test[-2000:]
    x_val = x_test[:-2000,:,:,:]
    y_val = y_test[:-2000]
    x_train = x_train
    y_train = y_train

    return x_val,y_val,x_sim,y_sim,x_train,y_train

def prep_data_mnist():
    #################### PREP DATA ##############################

    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()    #get data

    x_train = tf.keras.utils.normalize(tf.expand_dims(x_train, axis=3, name=None))
    x_test = tf.keras.utils.normalize(tf.expand_dims(x_test, axis=3, name=None))
    print(x_train.shape)

    x_sim = x_test[-2000:,:,:,:]
    y_sim = y_test[-2000:]
    x_val = x_test[:-2000,:,:,:]
    y_val = y_test[:-2000]
    x_train = x_train
    y_train = y_train

    return x_val,y_val,x_sim,y_sim,x_train,y_train

def prep_data_fashion():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

    x_train = tf.keras.utils.normalize(tf.expand_dims(x_train, axis=3, name=None))
    x_test = tf.keras.utils.normalize(tf.expand_dims(x_test, axis=3, name=None))
    print(x_train.shape)

    x_sim = x_test[-2000:,:,:,:]
    y_sim = y_test[-2000:]
    x_val = x_test[:-2000,:,:,:]
    y_val = y_test[:-2000]
    x_train = x_train
    y_train = y_train

    return x_val,y_val,x_sim,y_sim,x_train,y_train

### LENET - 5 - Edit ###
#def initModel_With_layers(layer_numbers):
#    [L1,L2,L3,L4,L5] = layer_numbers
#    ######### RE-DEF MODEL ############
#    shape=(28,28,1)
#    In = layers.Input(shape=shape,name="OnlyIP")
#    x1 = keras.layers.Conv2D(L1, kernel_size=5, strides=1,  activation='tanh', padding='same', name="Conv1_prune")(In)
#    x2 = keras.layers.AveragePooling2D()(x1)
#    x5 = keras.layers.Conv2D(L2, kernel_size=5, strides=1, activation='tanh', padding='valid', name="Conv2_prune")(x2)
#    x6 = keras.layers.Flatten()(x5)
#    x7 = keras.layers.Dense(L3, activation='tanh', name="Dense1_prune")(x6)
#    x7 = keras.layers.Dense(L4, activation='tanh', name="Dense2_prune")(x7)
#    x8 = keras.layers.Dense(L5, activation='softmax', name="NoDo")(x7)
#    model = Model([In],[x8], name="WOWOWOWOOW")
#    return model

### LENET-300 ###
#def initModel_With_layers(layer_numbers):
#    [L1,L2]=layer_numbers
#    ######### RE-DEF MODEL ############
#    shape=(28,28,1)
#    In = layers.Input(shape=shape,name="OnlyIP")
#    Flt = layers.Flatten()(In)
#    x1 = Dense(L1,activation='relu',name="Dense1_prune")(Flt)
#    x2 = Dense(L2,activation='relu',name="Dense2_prune")(x1)
#    #x3 = Dense(L3,name="dense3_prune")(x2)
#    x = Dense(10,activation='softmax',name="OPnodo")(x2)
#    model = Model([In],[x])
#    return model

########### COMPILE MODEL ###########

def compile_np_model(model):
    opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    model.compile(opt,'sparse_categorical_crossentropy',['accuracy','sparse_categorical_accuracy','sparse_top_k_categorical_accuracy'])
    return model

def compile_p_model(model):
    opt = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9)
    model.compile(opt,'sparse_categorical_crossentropy',['accuracy','sparse_categorical_accuracy','sparse_top_k_categorical_accuracy'])
    return model
    
def compile_p_rt_model(model):
    opt = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9)
    model.compile(opt,'sparse_categorical_crossentropy',['accuracy','sparse_categorical_accuracy','sparse_top_k_categorical_accuracy'])
    return model

def train_eval_save_np_model(epochs,base_dir,model_layers,conn):
    dst = load_imagenet("train")
    dsv = load_imagenet("validation")
    model = initModel_With_layers(model_layers)
    model = compile_np_model(model)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(base_dir+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)
    model.fit(dst,batch_size=256,epochs=epochs,callbacks=[model_checkpoint_callback])
    unpruned_val = model.evaluate(dsv)
    model.save(base_dir+"UnPrunedInitModel.h5")
    conn.send([unpruned_val,base_dir+"UnPrunedInitModel.h5"])
    conn.close()


def eval_save_np_model(epochs,base_dir,model_layers,conn):
    dsv = load_imagenet("validation")
    model = initModel_With_layers(model_layers)
    model = compile_np_model(model)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(base_dir+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)
    unpruned_val = model.evaluate(dsv)
    model.save(base_dir+"UnPrunedInitModel.h5")  
    conn.send([unpruned_val,base_dir+"UnPrunedInitModel.h5"])
    conn.close()
    

######### INITIAL SPASIRTY PRUNE ########

def train_p_model(init_model_loc,sparsity_pruning_epochs,prune_logdir,base_dir,conn):

    model = tensorflow.keras.models.load_model(init_model_loc)
    
    dst = load_imagenet("train")

    shape_tf = tf.data.experimental.cardinality(dst)
    first_el = shape_tf.numpy()
    end_step = first_el*sparsity_pruning_epochs

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

    model_for_pruning = compile_p_model(model_for_pruning)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+'model-prune{epoch:02d}.h5',
                                                                   save_weights_only=False,save_freq='epoch',
                                                                   save_best_only=False,
                                                                   verbose=1)
    model_for_pruning.fit(dst,
                          batch_size=256,
                          epochs=sparsity_pruning_epochs,
                          callbacks=[model_checkpoint_callback,tfmot.sparsity.keras.UpdatePruningStep(),tfmot.sparsity.keras.PruningSummaries(log_dir=prune_logdir),])

    model_for_pruning = strip_save_model(model_for_pruning,base_dir)

    conn.send([base_dir+'pruned_n_stripped_model.h5'])
    conn.close()

def strip_save_model(model_for_pruning,base_dir):

    ### SAVE SPARSLY PRUNED MODEL ###

    model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    save_param_file(model_for_pruning,base_dir)

    tf.keras.models.save_model(model_for_pruning, 
                               base_dir+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    return base_dir+'pruned_n_stripped_model.h5'

def save_param_file(model_for_pruning,base_dir):

    ### SAVE PARAMS TO FILE ####

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model_for_pruning.trainable_weights])
    with open(base_dir+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))

def compare_models(pruned_model_loc,conn):
    ##### Test and compare final models ######
    dsv = load_imagenet("validation")
    model_for_pruning = tensorflow.keras.models.load_model(pruned_model_loc)
    model_for_pruning = compile_np_model(model_for_pruning)
    sparse_pruned_val = model_for_pruning.evaluate(dsv)
    conn.send([sparse_pruned_val])
    conn.close()
                  
def find_neurons_2_prune(model_loc,model_layers,max_dev,layer_percent_consideration,Noise,sim_std,sim_mean,sim_shape,dataset_percentage,conn):

    dst = load_imagenet("train")

    model = tensorflow.keras.models.load_model(model_loc)

    layer_consideration = layer_percent_consideration/100

    ####### Find Neurons to prune ########

    i = 0
    submodels = []
    for layer in model.layers:
        if layer.name[-5:] == "prune":
            submodels.append(Model(inputs=model.input, outputs=model.layers[i].output))
        i+=1
    print("Calculating Dataset Stasticics")
    lowest_activations = []
    
    shape_tf = tf.data.experimental.cardinality(dst)
    first_el = shape_tf.numpy()
    samples_2_gen =round(first_el*dataset_percentage, -2)
    
    if sim_shape == None:
      
      ov = OnlineVariance(ddof=0)
      inputs = tfds.as_numpy(dst.map(lambda x,y: x))
      mean = tf.keras.metrics.Mean()
      i = 0
      
      for batch in tqdm(inputs):
          mean.update_state(batch)
          ov.include(np.std(batch))
          i+=1
          if i == samples_2_gen:
            break
             
      _,b,c,d = batch.shape
      sim_shape = (i,b,c,d)   
      sim_std = ov.std
      sim_mean = mean.result().numpy()

    if Noise:
        fake_data = np.random.normal(sim_mean, sim_std, size=(sim_shape))
        fake_data = np.interp(fake_data, (fake_data.min(), fake_data.max()), (-186.255, +214.7954))
    else:
        fake_data = x_sim

    for submodel in tqdm(submodels):
        if "Dense" in submodel.layers[-1].name:
            layer_units = submodel.layers[-1].units
            trim_ammount = int(layer_consideration*layer_units)
            if trim_ammount <= 0:
                trim_ammount = 1

            
            for h in range(int(samples_2_gen/100)):
                if h == 0:
                    submodel_prediction = np.absolute(submodel.predict(fake_data[h*100:(h+1)*100,:,:,:]))
                else:
                    submodel_prediction = np.vstack((submodel_prediction,np.absolute(submodel.predict(fake_data[h*100:(h+1)*100,:,:,:]))))
                    
            averaged_array = np.average(submodel_prediction,axis=0)

            lowest_10_values_location = np.argsort(averaged_array)[:trim_ammount]
            lowest_10_values = []
            
            for i in lowest_10_values_location:
                print(averaged_array[i])
                lowest_10_values.append(averaged_array[i])

            lowest_activations.append([lowest_10_values_location,lowest_10_values])

        if "Conv" in submodel.layers[-1].name:
            layer_units = submodel.layers[-1].filters
            trim_ammount = int(layer_consideration*layer_units)
            if trim_ammount <= 0:
                trim_ammount = 1
            
            for h in range(int(samples_2_gen/100)):
                if h == 0:
                    submodel_prediction = np.absolute(submodel.predict(fake_data[h*100:(h+1)*100,:,:,:]))
                else:
                    submodel_prediction = np.vstack((submodel_prediction,np.absolute(submodel.predict(fake_data[h*100:(h+1)*100,:,:,:]))))
            averaged_array = np.average(submodel_prediction,axis=0)
            
            interim = []
            for chanloc in range(averaged_array.shape[2]):
                interim.append(np.average(averaged_array[:,:,chanloc]))  # sum or avg

            averaged_array = interim
            lowest_10_values_location = np.argsort(averaged_array)[:trim_ammount]
            lowest_10_values = []

            for i in lowest_10_values_location:
                print(averaged_array[i])
                lowest_10_values.append(averaged_array[i])
            lowest_activations.append([lowest_10_values_location,lowest_10_values])

    to_remove = []
    act_vals = []
    for lowest_layer in lowest_activations:
        locs,acts = lowest_layer
        retain = 0
        for val in acts:
            if val > acts[0] + max_dev:
                break
            else:
                retain+=1
        rem_locs = list(locs[:retain])
        vals = acts[:retain]
        to_remove.append(rem_locs)
        act_vals.append(vals)
    

    for i in range(len(to_remove)):
        if len(to_remove[i]) == model_layers[i]:
            to_remove[i] = []
            act_vals[i] = []

    conn.send([to_remove,act_vals,sim_std,sim_mean,sim_shape])
    conn.close()

def remove_neurons_n_eval(model_loc,model_layers,to_remove,prune_loop,log_dir,conn):
    looping = True
    model = tensorflow.keras.models.load_model(model_loc)
    ########### Prune Neurons #################
    i=0
    surgeon = Surgeon(model)
    for layer in model.layers:
        if layer.name[-5:] == "prune":
            if len(to_remove[i]) != 0:
                surgeon.add_job('delete_channels', layer, channels=to_remove[i])
            i+=1

    new_model = surgeon.operate()

    model = new_model

    compile_np_model(model)
    dsv = load_imagenet("validation")
    pruned_val = model.evaluate(dsv)

    for i in range(len(to_remove)):
        model_layers[i] = model_layers[i] - len(to_remove[i])

    #save new model

    new_model_loc = log_dir+"/Neurons_Removed.h5"
    model.save(new_model_loc)

    conn.send([new_model_loc,model_layers,pruned_val,looping])
    conn.close()

def retrain(curr_model_loc,model_layers,retrain_epochs,prune_logdir,prune_loop,conn):
    model = tensorflow.keras.models.load_model(curr_model_loc)
    model = compile_np_model(model)
    ########### LOAD DATA ################
    dst = load_imagenet("train")
    dsv = load_imagenet("validation")
    ########### RETRAIN #################

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+str(prune_loop)+'/'+'model{epoch:02d}.h5',save_weights_only=False,save_freq='epoch',save_best_only=False ,verbose=1)
    model.fit(dst,batch_size=256, epochs=retrain_epochs,callbacks=[model_checkpoint_callback])
    retrained_val = model.evaluate(dsv)

    new_model_loc = prune_logdir+str(prune_loop)+"/Retrained"+str(prune_loop)+".h5"
    model.save(new_model_loc)

    conn.send([new_model_loc,model_layers,retrained_val])
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

    with open(prune_logdir+str(prune_loop)+'/'+'EvalResults.txt', 'w') as f:
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

def sparse_retrain(curr_model_loc,model_layers,sparsity_pruning_retrain_epochs,prune_logdir,prune_loop,dset_shape,conn):
    
    model = tensorflow.keras.models.load_model(curr_model_loc)
    model = compile_p_rt_model(model)

    ########### SPARSE_RETRAIN ##############
    dst = load_imagenet("train")

    end_step = dset_shape[0]*sparsity_pruning_retrain_epochs
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

    model_for_pruning = compile_p_rt_model(model_for_pruning)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(prune_logdir+str(prune_loop)+'/'+'model-prune{epoch:02d}.h5',
                                                                   save_weights_only=False,save_freq='epoch',
                                                                   save_best_only=False,
                                                                   verbose=1)
    model_for_pruning.fit(dst,
                          batch_size=256,
                          epochs=sparsity_pruning_retrain_epochs,
                          callbacks=[model_checkpoint_callback,tfmot.sparsity.keras.UpdatePruningStep(),tfmot.sparsity.keras.PruningSummaries(log_dir=prune_logdir+str(prune_loop)),])
    
    model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    tf.keras.models.save_model(model_for_pruning, 
                               prune_logdir+str(prune_loop)+'/'+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    model_for_pruning = compile_p_rt_model(model_for_pruning)

    conn.send([prune_logdir+str(prune_loop)+'/'+'pruned_n_stripped_model.h5',model_layers])
    conn.close()

def save_curr_model(curr_model_loc,model_layers,prune_logdir,prune_loop,conn):
    
    model = tensorflow.keras.models.load_model(curr_model_loc)
    model = compile_np_model(model)
    tf.keras.models.save_model(model, 
                               prune_logdir+str(prune_loop)+'/'+'pruned_n_stripped_model.h5', 
                               include_optimizer=False)

    conn.send([0])
    conn.close()

def save_num_of_params(curr_model_loc,model_layers,prune_loop,prune_logdir,conn):
    model = tensorflow.keras.models.load_model(curr_model_loc)
    ##### SAVE PARAMS #####
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    with open(prune_logdir+str(prune_loop)+'/'+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))
        
    conn.send([0])
    conn.close()

def get_flops(model_h5_path):
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops


def initModel_With_layers(model_layers):
    # input
    input = Input(shape =(224,224,3))
    # 1st Conv Block

    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu', name="Conv1_prune")(input)
    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu', name="Conv2_prune")(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 2nd Conv Block

    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu', name="Conv3_prune")(x)
    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu', name="Conv4_prune")(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 3rd Conv block

    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu', name="Conv5_prune")(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu', name="Conv6_prune")(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu', name="Conv7_prune")(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 4th Conv block

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu', name="Conv8_prune")(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu', name="Conv9_prune")(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu', name="Conv10_prune")(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 5th Conv block

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu', name="Conv11_prune")(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu', name="Conv12_prune")(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu', name="Conv13_prune")(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # Fully connected layers

    x = Flatten()(x)
    x = Dense(units = 4096, activation ='relu',name="Dense1")(x)
    x = Dense(units = 4096, activation ='relu',name="Dense2")(x)
    output = Dense(units = 1000, activation ='softmax',name="OPnodo")(x)
    # creating the model

    manual_VGG = Model (inputs=input, outputs =output)

    model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
    manual_VGG.set_weights(model.get_weights())

    return manual_VGG

def write_H_Params(base_dir,sparsity_pruning_epochs,retrain_epochs,max_dev,Global_Pruning,layer_percent_consideration):
    with open(base_dir+'/'+'Hyper_params.txt', 'w') as f: 
      f.write(str("sparsity_pruning_epochs:"))
      f.write(str(sparsity_pruning_epochs))
      f.write(str("\n retrain_epochs: \n"))
      f.write(str(retrain_epochs))
      f.write(str("\n max_dev:"))
      f.write(str(max_dev))
      f.write(str("\n Global_Pruning \n"))
      f.write(str(Global_Pruning))
      f.write(str("\n layer_percent_consideration \n"))
      f.write(str(layer_percent_consideration))
      f.write(str("\n compile_p_model \n"))
      f.write(str(0.00001))
      f.write(str("\n compile_p_rt_model \n"))
      f.write(str(0.00001))

if __name__ == "__main__":
    tensorflow.keras.backend.clear_session

    if platform == "linux" or platform == "linux2":
        print("Linux Detected")
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    elif platform == "darwin":
        sys.exit("OSx not supported")
    elif platform == "win32":
        print("Windows Detected")

    ########## INIT PARAMS #########
    
    model_layers = [64,64,128,128,256,256,256,512,512,512,512,512,512,4096,4096]
    initial_model_layers = copy.deepcopy(model_layers)
    epochs=1#2#20#40
    sparsity_pruning_epochs = 1
    retrain_epochs = 4
    sparsity_pruning_retrain_epochs = 1
    pruning_loops = 20
    max_dev = 0.0001  ### IMPORTANT !!!! max deviation from the lowest activation in a given layer
    layer_percent_consideration = 5 # percentage of neuron layer considered for pruning
    Global_Pruning = True
    Noise = True
    sim_std = None
    sim_mean = None
    sim_shape = None
    
    accu_drop_rt = 0.03 ### IMPORTANT !!!! accuracy drop allowed before retrain and stopping point
    accu_drop_end = 0.05
    dataset_percentage = 3/100

    base_dir =     '/data/scratch/eex869/Pruning_VGG_Just_Conv_6/'
    prune_logdir = base_dir + 'pruned/'
    finished_dir = base_dir + 'finished/'

    write_H_Params(base_dir,sparsity_pruning_epochs,retrain_epochs,max_dev,Global_Pruning,layer_percent_consideration)

    ##### SETUP ######
    
    parent_conn, child_conn = multiprocessing.Pipe()

    reader_process  = multiprocessing.Process(target=eval_save_np_model, args=(epochs,base_dir,model_layers,child_conn))
        
    reader_process.start()
        
    remove_points_returned = parent_conn.recv()
        
    unpruned_val_all,unpruned_init_model_loc = remove_points_returned

    reader_process.join()

    unpruned_val=unpruned_val_all[1]

    #### SAVE VALUE ####
    
    init_val = copy.deepcopy(unpruned_val)

    #### INIT PRUNE ####

    parent_conn, child_conn = multiprocessing.Pipe()

    reader_process  = multiprocessing.Process(target=train_p_model, args=(unpruned_init_model_loc,sparsity_pruning_epochs,prune_logdir,base_dir,child_conn))
        
    reader_process.start()
        
    remove_points_returned = parent_conn.recv()
        
    pruned_model_loc = remove_points_returned[0]

    reader_process.join()

    #### COMPARE MODELS ####

    parent_conn, child_conn = multiprocessing.Pipe()

    reader_process  = multiprocessing.Process(target=compare_models, args=(pruned_model_loc,child_conn))
        
    reader_process.start()
        
    remove_points_returned = parent_conn.recv()
        
    sparse_pruned_val_all = remove_points_returned[0]
    
    sparse_pruned_val = sparse_pruned_val_all[1]

    reader_process.join()

    ###loop
    prune_loop = 0
    looping = True
    
    curr_model_loc = pruned_model_loc
    
    while looping:

        tf.keras.backend.clear_session
        try:
            os.mkdir(prune_logdir+str(prune_loop))
        except:
            pass

        parent_conn, child_conn = multiprocessing.Pipe()

        reader_process  = multiprocessing.Process(target=find_neurons_2_prune, args=(curr_model_loc,model_layers,max_dev,layer_percent_consideration,Noise,sim_std,sim_mean,sim_shape,dataset_percentage,child_conn))
        
        reader_process.start()
        
        remove_points_returned = parent_conn.recv()
        
        remove_points, act_vals, sim_std,sim_mean,sim_shape = remove_points_returned

        reader_process.join()

        ### Compare activations between layers ####
        ### Global Pruning ###

        if Global_Pruning == True:
            min_acts = []
            for acts in act_vals:
                if acts != []:
                    min_acts.append(min(acts))
            min_act = min(min_acts)

            exclude = []
            for i in range(len(act_vals)):
                for j in range(len(act_vals[i])):
                    if min_act < act_vals[i][j] - max_dev:
                        exclude.append([i,j])

            exclude.reverse()
            if exclude != []:
                for i,j in exclude:
                    remove_points[i].pop(j)
        
        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=remove_neurons_n_eval, args=(curr_model_loc,model_layers,remove_points,prune_loop,prune_logdir,child_conn))
        
        reader_process.start()
        
        returned_vars = parent_conn.recv()
        
        curr_model_loc,model_layers,pruned_val_all,looping = returned_vars
        
        pruned_val = pruned_val_all[1]
        
        reader_process.join()

        if unpruned_val > pruned_val +  accu_drop_rt:

            parent_conn, child_conn = multiprocessing.Pipe()

            reader_process  = multiprocessing.Process(target=retrain, args=(curr_model_loc,model_layers,retrain_epochs,prune_logdir,prune_loop,child_conn))
        
            reader_process.start()
        
            returned_vars = parent_conn.recv()
        
            curr_model_loc,model_layers,retrained_val_all = returned_vars
            
            retrained_val = retrained_val_all[1]
        
            reader_process.join()

            prune_logdir+str(prune_loop)

            with open(prune_logdir+str(prune_loop)+'RT', 'w') as f:
                f.write(str('prune_occured'))
        else:
            retrained_val = unpruned_val
        
        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=compare_n_store, args=(unpruned_val,sparse_pruned_val,pruned_val,retrained_val,prune_logdir,prune_loop,child_conn))
        
        reader_process.start()
        
        returned_vars = parent_conn.recv()
        
        unpruned_val_w = returned_vars[0]
        
        reader_process.join()
                
        if unpruned_val > pruned_val +  accu_drop_rt:

            parent_conn, child_conn = multiprocessing.Pipe()
        
            reader_process  = multiprocessing.Process(target=sparse_retrain, args=(curr_model_loc,model_layers,sparsity_pruning_retrain_epochs,prune_logdir,prune_loop,sim_shape,child_conn))
        
            reader_process.start()
        
            returned_vars = parent_conn.recv()
        
            curr_model_loc,model_layers = returned_vars
        
            reader_process.join()

        else:

            parent_conn, child_conn = multiprocessing.Pipe()
        
            reader_process  = multiprocessing.Process(target=save_curr_model, args=(curr_model_loc,model_layers,prune_logdir,prune_loop,child_conn))
        
            reader_process.start()
        
            returned_vars = parent_conn.recv()
        
            not_used = returned_vars
        
            reader_process.join()
        
        parent_conn, child_conn = multiprocessing.Pipe()
        
        reader_process  = multiprocessing.Process(target=save_num_of_params, args=(curr_model_loc,model_layers,prune_loop,prune_logdir,child_conn))
        
        reader_process.start()
        
        returned_vars = parent_conn.recv()
        
        notused = returned_vars
        
        reader_process.join()
        
        with open(prune_logdir+str(prune_loop)+'/'+'model_layers.txt', 'w') as f:
            i = 0
            for layer in model_layers:
                f.write(str(model_layers[i])+'\n')
                i += 1

        unpruned_val = unpruned_val_w
        
        prune_loop += 1
        
        if init_val - accu_drop_end > retrained_val:
            print("Accuracy Dropped Too Far")
            break

    #final test
    #make dir
    os.mkdir(finished_dir)
    model = tensorflow.keras.models.load_model(curr_model_loc)
    #save model
    tf.keras.models.save_model(model, finished_dir+'/'+'finished_model.h5', include_optimizer=False)
    ##### SAVE PARAMS #####
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    with open(finished_dir+'/'+'model_params.txt', 'w') as f:
        f.write(str(trainableParams))
    #save layers
    with open(finished_dir+'/'+'model_layers.txt', 'w') as f:
        i = 0
        for layer in model.layers:
            if (layer.name[-5:] == "prune"):
                f.write(str(model_layers[i])+'\n')
                i += 1
    #save eval
    model = compile_np_model(model)
    dsv = load_imagenet("validation")
    finished = model.evaluate(dsv)

    final_flops = get_flops(finished_dir+'/'+'finished_model.h5')
    initial_flops = get_flops(base_dir+'/'+'UnPrunedInitModel.h5')

    with open(finished_dir+'/'+'EvalResults.txt', 'w') as f:
        f.write(str("Initial Model:"))
        f.write(str(init_val))
        f.write(str("\n Floops \n"))
        f.write(str(initial_flops))
        f.write(str("\nFinished model:"))
        f.write(str(finished))
        f.write(str("\n Floops \n"))
        f.write(str(final_flops))