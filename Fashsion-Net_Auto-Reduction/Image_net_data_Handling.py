import tensorflow as tf
import tensorflow_datasets as tfds

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

#load_imagenet(section) section = "train" or "validation"