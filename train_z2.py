import os
import scipy.misc
import numpy as np

from forward_model import DCGAN
from utils import pp, visualize, to_json, show_all_variables
from glob import glob
from ops import *
from utils import *

import tensorflow as tf
import matplotlib.pyplot as plt

%matplotlib inline

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

pp.pprint(flags.FLAGS.__flags)

if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
  dcgan = DCGAN(
      sess,
      input_width=FLAGS.input_width,
      input_height=FLAGS.input_height,
      output_width=FLAGS.output_width,
      output_height=FLAGS.output_height,
      batch_size=FLAGS.batch_size,
      sample_num=FLAGS.batch_size,
      dataset_name=FLAGS.dataset,
      input_fname_pattern=FLAGS.input_fname_pattern,
      crop=FLAGS.crop,
      checkpoint_dir=FLAGS.checkpoint_dir,
      sample_dir=FLAGS.sample_dir)

show_all_variables()

if not dcgan.load(FLAGS.checkpoint_dir)[0]:
  raise Exception("[!] Train a model first, then run test mode")

data = glob("./data/celebA/*.jpg")


sample_files = data[0:64] #change to 64 images
sample = [get_image(sample_file, input_height=dcgan.input_height,
                    input_width=dcgan.input_width,
                    resize_height=dcgan.output_height,
                    resize_width=dcgan.output_width,
                    crop=dcgan.crop,
                    grayscale=dcgan.grayscale) for sample_file in sample_files]
v = np.reshape(sample, (64, 64*64*3))
n = v.shape[1]
m = 1000
A = np.random.randn(n, m).astype('float32')
y = np.dot(v, A) 

y_placeholder = tf.placeholder(tf.float32,[None,m])
our_loss = tf.reduce_mean( tf.reduce_sum( (tf.matmul(tf.reshape(dcgan.G, [64, -1]) , A) - y_placeholder)**2 ) ) 
#z_optim = tf.train.AdamOptimizer(0.0002, 0.5).minimize(our_loss, var_list=dcgan.z)
    
grad = tf.gradients(our_loss, dcgan.z)
z_0 = np.random.uniform(-0.5, 0.5, size=(64 , 100))

prev2 = 9999999999
prev1 = 999999999
rate = 0.002
count = 0

errs = []
    
with tf.Session() as sess:
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    
    while prev1 < prev2 or rate > 0.0000000002: # run until error stops decreasing or reaches threshhold, then print result at that point
        print('iteration {}'.format(count))
        a, closs, b = sess.run([grad, our_loss, dcgan.G],
            feed_dict={ 
              dcgan.z: z_0,
              y_placeholder: y
            })
        count += 1
        prev2, prev1 = prev1, closs
        if count % 100 == 0:
            rate /= 2
        print(closs)
        errs.append(closs)
        z_0 = z_0 - rate*a[0]
print(a[0])
plt.imshow(b[0])

      
