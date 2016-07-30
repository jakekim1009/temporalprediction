import os
import scipy.misc
import numpy as np
import random
from utils import pp
import utils
import test

from model import PGN
from data.Bouncing_balls_data_reader import Bouncing_Balls_Data_Reader


import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train (you can always interrupt) [500]")

#flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
#flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_string("dataset", "/atlas/u/anthony/future_frame_prediction/data/Bouncing_balls_data_color", "The path of the dataset")
flags.DEFINE_string("dataset_name", "debug_bouncing_balls", "The name of the dataset when saved to checkpoints")
flags.DEFINE_string("checkpoint_dir", "checkpoints/unlabeled_checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples/unlabeled_samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_test", False, "True for testings, false to not run testing [False]")
flags.DEFINE_boolean("is_debug", False, "True for debugging (uses small dataset) [False]")
flags.DEFINE_boolean("is_visualize", False, "True for visualizing [False]")
flags.DEFINE_float("lambda_adv_loss", 0.0002, "The weight of the adverserial loss during training [0.0002]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    random.seed(31241)
    np.random.seed(41982)
    tf.set_random_seed(1327634)

    color = True # Must change this and the dataset Flags to the correct path to use color
    if FLAGS.is_debug:
       reader = Bouncing_Balls_Data_Reader(FLAGS.dataset, FLAGS.batch_size, color=color, train_size=160*5, validation_size=8*5, test_size=8*5, num_partitions=5)
    else:
       reader = Bouncing_Balls_Data_Reader(FLAGS.dataset, FLAGS.batch_size, color=color)

    data_fn = lambda epoch, batch_index: reader.read_data(batch_index, reader.TRAIN)
    frame_shape = reader.read_data(0, reader.TRAIN).shape[2:]
    print("Frame shape: ", frame_shape)
    num_batches = reader.num_batches(reader.TRAIN)
    print("Num batches: %d" % num_batches)
    input_sequence_range = range(5, 16)
    print("Input sequence range min: %d, max: %d" % (min(input_sequence_range), max(input_sequence_range)))

    save_sample_fn = utils.gen_save_sample_fn(FLAGS.sample_dir, image_prefix="train")

    with tf.Session() as sess:
        pgn  = PGN(sess, FLAGS.dataset_name, FLAGS.epoch, num_batches, FLAGS.batch_size, input_sequence_range,
                 data_fn, frame_shape=frame_shape, save_sample_fn=save_sample_fn, checkpoint_dir=FLAGS.checkpoint_dir,
                 lambda_adv_loss= FLAGS.lambda_adv_loss)

        if FLAGS.is_train:
            pgn.train()
        else:
            print("Loading from: %s" %(FLAGS.checkpoint_dir,))
            if pgn.load(FLAGS.checkpoint_dir) :
               print(" [*] Successfully loaded")
            else:
               print(" [!] Load failed")

        if FLAGS.is_test:
           result = test.test(pgn, reader)
           result_str = pp.pformat(result)
           fid = open(os.path.join(FLAGS.sample_dir, 'test_out.txt'), mode='w')
           fid.write(unicode(result_str))
           fid.close()

        if FLAGS.is_visualize:
           for i in range(3):
               vid_seq = reader.read_data(i, data_set_type=reader.TEST, batch_size=1)[:, 0, :, :, :]
               utils.make_prediction_gif(pgn, os.path.join(FLAGS.sample_dir, 'vis_%d.gif' % i), video_sequence=vid_seq)
           utils.plot_convergence(pgn.get_MSE_history(), "MSE Convergence",
                        path=os.path.join(FLAGS.sample_dir, "vis_MSE_convergence.png"))

if __name__ == '__main__':
    tf.app.run()