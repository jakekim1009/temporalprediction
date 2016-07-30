"""
Save and load code as well as general structure adapted from https://github.com/carpedm20/DCGAN-tensorflow
"""
import os
import sequence_ops as seq_ops
import static_ops as sta_ops
import numpy as np
import tensorflow as tf
import time

"""
We might want to move things around, but the building blocks are here
"""

def encoder(video_sequence, name, reuse=False, train=True):
    """
    Encodes each frame in the sequence
    video_sequence has size [num_frames, num_batchs, image_height, image_width, channels]
    the output as size [num_frames, num_batches, hidden_variable_length]

    Note that the output is not the result of a fully connected layer, it is the result
    of a convolution followed by a reshaping - some spatial information is lost, but the
    subsequent parts of the code do not use them anyway
    TODO clean up the hardcoded values below
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
       h0 = seq_ops.conv2d(video_sequence, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='h0_conv')
       h0 = seq_ops.max_pool(tf.nn.relu(h0), k_h = 2, k_w=2, d_h=2, d_w=2)
       h1 = seq_ops.conv2d(h0, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='h1_conv')
       h1 = seq_ops.max_pool(tf.nn.relu(h1), k_h=2, k_w=2, d_h=2, d_w=2)

       h1_shape = h1.get_shape().as_list()
       print("Encoder output before reshape shape (num_frames, num_batches, h, w, c): ", h1_shape)
       return tf.reshape(h1, [h1_shape[0], h1_shape[1], -1])
    # use the improved GAN version of batch normalization if you use batch normalization

def conv_LSTM(input_sequence, input_sequence_length, name, num_hidden_units, reuse=False, train=True):
    """
    Same as the LSTM but input_sequence has size [num_frames, num_batches, h, w, latent_length]
    and the output has size [num_batches, h, w, latent_length]
    http://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
    shows that deeper conv layers result in better performance with fewer parameters
    """
    assert num_hidden_units == 128 # Make sure you want something different before you change this

    def conv_helper(input_1, input_2, name):
           return sta_ops.conv2d_2(input_1, input_2, num_hidden_units, k_h=3, k_w=3, d_h=1, d_w=1, name=name)

    def LSTM_step(prev_cell, prev_hidden, current_input):
        input_gate = tf.sigmoid(conv_helper(current_input, prev_hidden, 'input_gate'))
        forget_gate = tf.sigmoid(conv_helper(current_input, prev_hidden, 'forget_gate'))
        current_cell = tf.mul(forget_gate, prev_cell) + tf.mul(input_gate,
                               tf.tanh(conv_helper(current_input, prev_hidden, 'memory_cell')))
        out_gate = tf.sigmoid(conv_helper(current_input, prev_hidden, 'out_gate'))
        current_hidden = tf.mul(out_gate, tf.tanh(current_cell))
        return current_cell, current_hidden

    with tf.variable_scope(name, reuse=reuse) as scope:
       num_frames, batch_size, h, w, c  = input_sequence.get_shape()
       init_cell = tf.zeros([batch_size, h, w, num_hidden_units])
       init_hidden = tf.zeros([batch_size, h, w, num_hidden_units])

       hidden_outputs = [init_hidden]
       cell_outputs = [init_cell]
       split_input_sequence = tf.split(0, num_frames, input_sequence)
       for input_slice in split_input_sequence:
           input_slice = tf.squeeze(input_slice, squeeze_dims=[0])
           next_cell, next_hidden = LSTM_step(cell_outputs[-1], hidden_outputs[-1], input_slice)
           cell_outputs.append(next_cell)
           hidden_outputs.append(next_hidden)
           scope.reuse_variables()

    with tf.variable_scope(name, reuse=reuse) as scope:
       # concatenate hidden units across time, shape is now [num_frames, num_batches, h, w, num_hidden_units]
       hidden_outputs = tf.concat(0, [tf.expand_dims(hidden, 0) for hidden in hidden_outputs])
       print("LSTM hidden_outputs shape (num_frames + 1, num_batches, h, w, num_hidden_units): ", hidden_outputs.get_shape().as_list())
       # keep in mind hidden_outputs[0] is the initial hidden output
       _, num_batches, h, w, num_hidden_units = hidden_outputs.get_shape().as_list() # We do this instead of -1 to avoid None in the shape
       hidden_output_after_input_sequence = tf.slice(hidden_outputs, [input_sequence_length,0,0,0,0], [1, num_batches, h, w, num_hidden_units])
       print("LSTM final output before squeeze shape (1, num_batches, h, w, num_hidden_units): ",
                                                              hidden_output_after_input_sequence.get_shape().as_list())
       hidden_output_after_input_sequence = tf.squeeze( hidden_output_after_input_sequence, squeeze_dims=[0])
       print("LSTM final output shape (num_batches, h, w, num_hidden_units): ", hidden_output_after_input_sequence.get_shape().as_list())
       return hidden_output_after_input_sequence

def LSTM(input_sequence, input_sequence_length, name, num_hidden_units, reuse=False, train=True):
    """
    input_sequence has size [num_frames, num_batches, input_variable_length]
    the output has size [num_batches, latent_variable_length]
    stopping_point should be a tf.Variable or placeholder that is not trainable that
        has its value equal to the length of the input sequence

    TODO
    The current implementation unrolls the recurrence for the maximimum possible sequence length
    When you unroll too much you get an out of memory error (e.g. the Bouncing Balls
    dataset with 15 frames in the input gives an out of memory error)  perhaps methods
    that do not unroll are more memory efficient
    Using the control flow capabilities of tensorflow it may be possible to avoid unrolling the entire loop
    and instead dynamically do the unrolling to the input sequence length
    It may also be possible to not unroll entirely if you use the LSTMCell and dynamic rnn classes,
    although they are not as transparent as I would like
    """
    # TODO As we deviate from the paper, we may want to just use the LSTMCell and RNNMultiCell functions
    # LSTMCell is probably identical to the below and RNNMultiCell lets you do deep recurrent networks
    nhu = num_hidden_units

    print("LSTM input shape (num_frames, num_batches, input_variable_length): ", input_sequence.get_shape().as_list())

    # Start simple Tensor flow LSTM implementation
    with tf.variable_scope(name, reuse=reuse) as scope:
        cell = tf.nn.rnn_cell.LSTMCell(nhu, state_is_tuple=True)
        batch_size = input_sequence.get_shape().as_list()[1]
        sequence_length_tiled = tf.fill([batch_size], input_sequence_length)
        init_state = cell.zero_state(batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, input_sequence,
                       initial_state=init_state, sequence_length=sequence_length_tiled, time_major=True)

        _, num_batches, num_hidden_units = outputs.get_shape().as_list() # We do this instead of -1 to avoid None in the shape
        output_after_input_sequence = tf.slice(outputs, [input_sequence_length-1,0,0], [1, num_batches, num_hidden_units])
        output_after_input_sequence = tf.squeeze( output_after_input_sequence, squeeze_dims=[0])
        return output_after_input_sequence
    # End simple tensor flow LSTM implementation

    """
    Personal LSTM implementation (also works)
    def LSTM_step(prev_cell, prev_hidden, current_input):
        input_gate = tf.sigmoid(sta_ops.linear_2(current_input, prev_hidden, nhu, scope='input_gate'))
        forget_gate = tf.sigmoid(sta_ops.linear_2(current_input, prev_hidden, nhu, scope='forget_gate'))
        current_cell = tf.mul(forget_gate, prev_cell) + tf.mul(input_gate,
                               tf.tanh(sta_ops.linear_2(current_input, prev_hidden, nhu, scope='memory_cell')))
        out_gate = tf.sigmoid(sta_ops.linear_2(current_input, prev_hidden, nhu, scope='out_gate'))
        current_hidden = tf.mul(out_gate, tf.tanh(current_cell))
        return current_cell, current_hidden
    with tf.variable_scope(name, reuse=reuse) as scope:
       num_frames, batch_size, input_dim = input_sequence.get_shape()
       init_cell = tf.zeros([batch_size, num_hidden_units])
       init_hidden = tf.zeros([batch_size, num_hidden_units])

       hidden_outputs = [init_hidden]
       cell_outputs = [init_cell]
       split_input_sequence = tf.split(0, num_frames, input_sequence)
       for input_slice in split_input_sequence:
           input_slice = tf.squeeze(input_slice, squeeze_dims=[0])
           next_cell, next_hidden = LSTM_step(cell_outputs[-1], hidden_outputs[-1], input_slice)
           cell_outputs.append(next_cell)
           hidden_outputs.append(next_hidden)
           scope.reuse_variables()
    # reset the reuse for subsequent variables
    with tf.variable_scope(name, reuse=reuse) as scope:
       # concatenate hidden units across time, shape is now [num_frames, num_batches, num_hidden_units]
       hidden_outputs = tf.concat(0, [tf.expand_dims(hidden, 0) for hidden in hidden_outputs])
       print("LSTM hidden_outputs shape (num_frames + 1, num_batches, num_hidden_units): ", hidden_outputs.get_shape().as_list())
       # keep in mind hidden_outputs[0] is the initial hidden output
       _, num_batches, num_hidden_units = hidden_outputs.get_shape().as_list() # We do this instead of -1 to avoid None in the shape
       hidden_output_after_input_sequence = tf.slice(hidden_outputs, [input_sequence_length,0,0], [1, num_batches, num_hidden_units])
       print("LSTM final output before squeeze shape (1, num_batches, num_hidden_units): ", hidden_output_after_input_sequence.get_shape().as_list())
       hidden_output_after_input_sequence = tf.squeeze( hidden_output_after_input_sequence, squeeze_dims=[0])
       print("LSTM final output shape (num_batches, num_hidden_units): ", hidden_output_after_input_sequence.get_shape().as_list())
       return hidden_output_after_input_sequence
    """

def decoder(latent_variables, image_height, image_width, image_channels, name, reuse=False, train=True):
    """
    latent_variables has size [num_batches, latent_variable_length]
    the output has size [num_batches, image_height, image_width, channels]

    TODO clean up hardcoded values below
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
       num_batches, num_latent_variables = latent_variables.get_shape().as_list()
       # TODO fully connected layer for more complex experiments
       # Note that this reshaping can fail if the dimensions do not line up
       reshaped_latent_variables = tf.reshape(latent_variables, [num_batches, image_height/4, image_width/4, -1])
       h0 = tf.image.resize_images(reshaped_latent_variables, image_height/2, image_width/2,
                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
       h0 = tf.nn.relu(sta_ops.conv2d(h0, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='h0_conv'))
       h1 = tf.image.resize_images(h0, image_height, image_width,
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
       h1 = tf.nn.tanh(sta_ops.conv2d(h1, image_channels, k_h=3, k_w=3, d_h=1, d_w=1, name='h1_conv'))
       return h1

def generator(video_sequence, input_sequence_length, num_LSTM_units, name="gen", max_pixel_val=255.0, reuse=False, train=True):
    """
    Encodes each frame in the sequence
    video_sequence has size [num_frames, num_batchs, image_height, image_width, channels]
    the output has size [num_batches, image_height, image_width, channels]
    num_frames is the maximum number of frames in a sequence, less than the maximum can be used
    by specifying input_sequence_length

    input_sequence_length is a non-trainable variable or placeholder that specifies the number of input frames to use
    """
    # TODO idk how the paper scales image ranges,
    # but it would probably be a good idea to convert the images to have range [-1, 1]
    # and then unconvert back at the end, check what the paper does
    with tf.variable_scope(name, reuse=reuse) as scope:
        video_sequence = (2.0*video_sequence)/max_pixel_val - 1
        num_frames, num_batches, image_height, image_width, image_channels = video_sequence.get_shape().as_list()
        encoded = encoder(video_sequence, "encoder", reuse=reuse, train=train)
        lstm_output = LSTM(encoded, input_sequence_length, "lstm", num_LSTM_units, reuse=reuse, train=train)
        decoded = decoder(lstm_output, image_height, image_width, image_channels, "decoder", reuse=reuse, train=train)
        return ((decoded + 1) * max_pixel_val) / 2.0, lstm_output, encoded

def discriminator_final_frame_encoder(final_frame, name, num_LSTM_units, reuse=False, train=True):
    print("Final_frame_encoder final_frame shape: ", final_frame.get_shape().as_list())

    with tf.variable_scope(name, reuse=reuse) as scope:
        final_frame_expanded = tf.expand_dims(final_frame, 0)
        encoded = encoder(final_frame_expanded, 'encoder', reuse=reuse, train=train)
        encoded = tf.squeeze(encoded, squeeze_dims=[0])
        encoded = tf.nn.tanh(sta_ops.linear(encoded, num_LSTM_units, scope="final_encoder_FC_layer"))
        return encoded

def discriminator_MLP(input_, name, reuse=False, train=True):
    """
    Input is [batch_size, LSTM_size*2]
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
         batch_size, latent_size = input_.get_shape().as_list()
         h0 = sta_ops.lrelu(sta_ops.linear(input_,latent_size, scope="MLP_h0_linear"))
         h1 = sta_ops.lrelu(sta_ops.linear(h0, latent_size, scope="MLP_h1_linear"))
         h2 = sta_ops.linear(h1, 1, scope="MLP_h2_linear")
         return tf.nn.sigmoid(h2), h2

def discriminator(video_sequence, input_sequence_length, final_frame, num_LSTM_units, name="discrim", max_pixel_val=255.0, reuse=False, train=True):
    """
    final_frame has shape [batch_size, h, w, c]
    """
    # TODO params explanation
    print("Discriminator video_sequence shape: ", video_sequence.get_shape().as_list())

    with tf.variable_scope(name, reuse=reuse) as scope:
        video_sequence = (2.0*video_sequence)/max_pixel_val - 1
        final_frame = (2.0 * final_frame) / max_pixel_val - 1
        num_frames, num_batches, image_height, image_width, image_channels = video_sequence.get_shape().as_list()

        encoded = encoder(video_sequence, "encoder", reuse=reuse, train=train)
        lstm_output = LSTM(encoded, input_sequence_length, "lstm", num_LSTM_units, reuse=reuse, train=train)

        final_frame_encoder = discriminator_final_frame_encoder(final_frame, "final_frame_encoder",
                                                                num_LSTM_units, reuse=reuse, train=train)

        MLP_input = tf.concat(1, (lstm_output, final_frame_encoder))
        p_real, logits = discriminator_MLP(MLP_input, "MLP", reuse=reuse, train=train)

        return p_real, logits

class PGN(object):

    # TODO paper was unclear about learning rate decay and momentum
    # If batch norm is ever used, you may need to have separate code for the sampler

    def __init__(self, sess, dataset_name, num_epochs, num_batches, batch_size, sequence_length_range,
                 data_fn, save_sample_fn=None, checkpoint_dir=None, max_pixel_value=1, frame_shape=[30,30,1],
                 num_LSTM_units = 1568,
                 generator_learning_rate=0.001, generator_decay=0.9, generator_momentum=0.0,
                 lambda_adv_loss=0.0002, discriminator_learning_rate=0.01, discriminator_momentum=0.5, discriminator_decay=0.0):
        """
        TODO some of these parameters should be parameters to the train function and not be here
        sequence_length_range: The possible lengths to use as input to the model, should be a list of values e.g. [3,5,6,7,12]
                               To specify one value, just do [val] e.g. [5]
                               The input length is choosen uniformly from this list at the start of each epoch, you can change the distribution
                               by including a value multiple times e.g. [5,5,5,6,7,12,12]
        data_fn:  A function used to get the data to feed to the model. data_fn(epoch, batch_index) should return a batch of videos
                  of the shape [self.max_sequence_length, self.batch_size, image_height, image_width, channels]
        save_sample_fn: A function that accepts 3 parameters and saves the output.  Parameters are
                        save_sample_fn(epoch, batch_index, input_sequence, ground_truth, prediction)
                        input_sequence has dimension [num_frames, num_batch, h, w, c]
                        ground_truth and prediction are lists, each element has dimension [num_batch, h, w, c] and they are successive frames
                        If None, does not save samples
        """

        self.sess = sess
        self.dataset_name = dataset_name
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.sequence_length_range = sequence_length_range
        self.max_sequence_length = max(sequence_length_range) + 1 # We need to store the ground truth frames we are trying to predict here too
        self.data_fn = data_fn
        self.save_sample_fn = save_sample_fn
        self.frame_shape = list(frame_shape) # to convert tuples to lists
        self.max_pixel_value = max_pixel_value
        self.checkpoint_dir = checkpoint_dir

        self.gen_lr = generator_learning_rate
        self.gen_decay = generator_decay
        self.gen_momentum = generator_momentum

        assert lambda_adv_loss <= 1.0
        assert lambda_adv_loss >= 0.0
        self.lambda_adv_loss = lambda_adv_loss
        self.discrim_lr = discriminator_learning_rate
        self.discrim_momentum = discriminator_momentum
        self.discrim_decay = discriminator_decay

        self.LSTM_units = num_LSTM_units

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        num_LSTM_units = self.LSTM_units
        # place the entire video sequence in the placeholder, and the ground truth will be the frame after
        # the input sequence self.video_sequence[self.input_sequence_length]
        self.video_sequence = tf.placeholder(tf.float32, [self.max_sequence_length, self.batch_size] + self.frame_shape)
        self.input_sequence_length = tf.placeholder(tf.int32, [])
        self.input_sequence = tf.slice(self.video_sequence, [0,0,0,0,0], [self.input_sequence_length, -1, -1, -1, -1])
        self.ground_truth = tf.squeeze(tf.slice(self.video_sequence, [self.input_sequence_length,0,0,0,0],
                                                [1] + self.video_sequence.get_shape().as_list()[1:]), squeeze_dims=[0])

        self.generator, self.gen_lstm_output, self.gen_ecoder_output = generator(self.video_sequence, self.input_sequence_length,
                                   num_LSTM_units, max_pixel_val=self.max_pixel_value, reuse=False, train=True)
        generator_variables = [var for var in tf.trainable_variables() if "gen" in var.name]
        print("Generator Variables:")
        print([var.name for var in generator_variables])

        discriminator_real, discriminator_real_logits = discriminator(self.video_sequence,
                                    self.input_sequence_length, self.ground_truth, num_LSTM_units,
                                    max_pixel_val=self.max_pixel_value, reuse=False, train=True)
        discriminator_fake, discriminator_fake_logits = discriminator(self.video_sequence,
                                    self.input_sequence_length, self.generator, num_LSTM_units,
                                    max_pixel_val=self.max_pixel_value, reuse=True, train=True)
        discriminator_variables = [var for var in tf.trainable_variables() if "discrim" in var.name]
        print("Discriminator Variables:")
        print([var.name for var in discriminator_variables])

        # loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(discriminator_real_logits,
                                                                             tf.ones_like(discriminator_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(discriminator_fake_logits,
                                                                             tf.zeros_like(discriminator_fake)))
        d_loss = 0.5*d_loss_real + 0.5*d_loss_fake

        self.MSE = tf.reduce_mean( tf.pow( self.ground_truth - self.generator, 2) )
        g_loss = self.MSE
        if self.lambda_adv_loss > 1e-10: # if lambda > 0
           adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(discriminator_fake_logits,
                          tf.ones_like(discriminator_fake)))
           g_loss = self.MSE + self.lambda_adv_loss * adv_loss

        # optimizer
        self.g_optim = tf.train.RMSPropOptimizer( self.gen_lr, decay=self.gen_decay,
                      momentum=self.gen_momentum).minimize(g_loss, var_list=generator_variables)

        d_learning_rate_with_decay = self.discrim_lr
        self.discrim_decay_iterations = tf.placeholder(tf.int32, [])
        if not self.discrim_decay == 0.0:
           print("Using decay %4.10f in discriminator learning rate"  % self.discrim_decay)
           d_learning_rate_with_decay = self.discrim_lr * (1. / (1. + self.discrim_decay * self.discrim_decay_iterations))
           # This is not exponential decay, but its what the Kera source code does

        self.d_optim = tf.train.MomentumOptimizer(d_learning_rate_with_decay, self.discrim_momentum) \
                        .minimize(d_loss, var_list=discriminator_variables)
        # Below are variables that are updated manually

        self.epoch = tf.get_variable("Epoch", shape=[], dtype=tf.int32,
                     initializer=tf.constant_initializer(0), trainable=False)
        self.batch_index = tf.get_variable("Batch_Index", shape=[], dtype=tf.int32,
                     initializer=tf.constant_initializer(0), trainable=False)

    def set_epoch(self, val):
        self.sess.run(self.epoch.assign(val))

    def set_batch_index(self, val):
        self.sess.run(self.batch_index.assign(val))

    def update_MSE_history(self, counter, MSE, save):
        path = os.path.join(self.checkpoint_dir, "MSE_history_data.npy")
        if not hasattr(self, 'MSE_history_data'):
           if os.path.exists(path):
              self.MSE_history_data = np.load(path)
           else:
              self.MSE_history_data = np.zeros((0,2))
        self.MSE_history_data = np.concatenate((self.MSE_history_data, [[counter,MSE]]), axis=0)
        if save:
           np.save(path, self.MSE_history_data)

    def get_MSE_history(self):
        path = os.path.join(self.checkpoint_dir, "MSE_history_data.npy")
        if not os.path.exists(path):
           raise Exception("Could not load MSE history, file not found")
        return np.load(path)

    def train(self):

        if self.checkpoint_dir is not None and self.load(self.checkpoint_dir): # Load
           print(" [*] Checkpoint load success")
        else:
           print(" [!] could not load model from %s" % self.checkpoint_dir)
           self.sess.run(tf.initialize_all_variables())

        start_time = time.time()
        current_epoch = self.epoch.eval()
        current_batch = self.batch_index.eval()
        print("Starting from epoch %d and batch %d" % (current_epoch, current_batch))
        while current_epoch < self.num_epochs:
              epoch_input_length = np.random.permutation(self.sequence_length_range)[0]

              while current_batch < self.num_batches:

                  original_batch = self.data_fn(current_epoch, current_batch)
                  batch = original_batch[:self.max_sequence_length, :, :, :, :]
                  assert np.all(batch <= self.max_pixel_value)
                  assert np.all(batch >= 0)
                  # It would be wise to come up with a better check that the images are correctly normalized to [-1, 1]

                  feed_dict = {self.video_sequence: batch, self.input_sequence_length: epoch_input_length,
                               self.discrim_decay_iterations: current_epoch}

                  if self.lambda_adv_loss > 1e-10: # if lambda > 0
                     self.sess.run(self.d_optim, feed_dict=feed_dict)

                  self.sess.run(self.g_optim, feed_dict=feed_dict)

                  current_batch += 1
                  self.set_batch_index(current_batch)

                  counter = current_epoch*self.num_batches + current_batch
                  if np.mod(counter, 5) == 1:
                     print("Epoch[%d/%d], Batch[%d/%d], time: %d" % (current_epoch, self.num_epochs,
                            current_batch, self.num_batches, time.time() - start_time ))

                     if not self.lambda_adv_loss > 1e-10:
                        print("Not running discriminator, lambda_adv_loss is zero or too small")

                  if np.mod(counter, 20) == 1:
                     err_MSE = self.sess.run(self.MSE, feed_dict=feed_dict)

                     print("MSE: %4.4f" %(err_MSE,))
                     self.update_MSE_history(counter, err_MSE, np.mod(counter, 500) == 1)

                  if np.mod(counter, 100) == 1:
                     if self.save_sample_fn is not None:
                        input_seq, gt_frame, prediction, lstm_out, encoder_out = self.sess.run([self.input_sequence,
                                                      self.ground_truth, self.generator, self.gen_lstm_output,
                                                      self.gen_ecoder_output], feed_dict=feed_dict)

                        print("gen min: %4.4f, max: %4.4f, LSTM_output min: %4.4f, max: %4.4f, num_zero: %d, Encoder min: %4.4f, max: %4.4f" \
                               % (np.min(prediction), np.max(prediction),
                                  np.min(lstm_out), np.max(lstm_out), np.sum(np.abs(lstm_out) < 0.000001),
                                  np.min(encoder_out), np.max(encoder_out)))

                        sample_next_frames = 10
                        prediction = self.predict_next_frames(original_batch[:10, 0, :, :, :], num_frames=sample_next_frames)
                        gt_frame = original_batch[10:20, 0, :, :, :]
                        # after the split the frame dimension becomes the batch dimension, which was removed, so this is ok
                        self.save_sample_fn(current_epoch, current_batch, input_seq,
                               np.split(gt_frame, sample_next_frames), np.split(prediction, sample_next_frames))
                     else:
                        print("Sample save function not specified, cannot save samples")
                  if np.mod(counter, 500) ==  1:
                     if self.checkpoint_dir is not None:
                        print("Saving checkpoint")
                        self.save(self.checkpoint_dir, counter)
                     else:
                        print("Could not save checkpoint")
                     # ...

              current_batch = 0
              current_epoch += 1
              self.set_epoch(current_epoch)
              self.set_batch_index(current_batch)

        counter = current_epoch*self.num_batches + current_batch
        self.save(self.checkpoint_dir, counter)

    def batch_predict_next_frame(self, input_video_sequence):
        # TODO
        pass

    def predict_next_frame(self, input_video_sequence):
        """
        input_video_sequence is a numpy array with shape [num_frames, h, w, c]
        num_frames must be less than or equal to the maximum of the sequence_length_range
        parameter passed at the creation of the PGN
        output is one frame and has the same shape as a single frame [h, w, c]
        """
        if not hasattr(self, "batch_size_one_sampler"):
            self.batch_size_one_video_sequence = tf.placeholder(tf.float32, [self.max_sequence_length, 1] + self.frame_shape)
            self.batch_size_one_sampler, _, _, = generator(self.batch_size_one_video_sequence, self.input_sequence_length,
                                   self.LSTM_units, max_pixel_val=self.max_pixel_value, reuse=True, train=False)

        num_frames, h, w, c = input_video_sequence.shape
        assert num_frames < self.max_sequence_length # should be < not <= because max_sequence_length includes ground truth
        input_ = np.zeros(self.batch_size_one_video_sequence.get_shape().as_list())
        input_[:num_frames, 0, :, :, :] = input_video_sequence

        feed_dict = {self.batch_size_one_video_sequence: input_, self.input_sequence_length: num_frames}
        output = self.sess.run(self.batch_size_one_sampler, feed_dict=feed_dict)
        return output[0, :, :, :]



    def predict_next_frames(self, input_video_sequence, num_frames=10, front_pad=0, back_pad=0):
        """
        input_video_sequence is a numpy array with shape [num_frames, h, w, c]
        num_frames must be less than or equal to the maximum of the sequence_length_range
        parameter passed at the creation of the PGN
        The number of frames fed as input to the model will be consistent
        i.e. the model will always be fed X frames where X is the number of frames in input_video_sequence
        front_pad number of blank frames will be added to the beginning of the sequence
        back_pad number of blank frames will be added to the end of the sequence
        """
        output = np.zeros([num_frames + back_pad] + self.frame_shape)
        input_sequence_length, h, w, c = input_video_sequence.shape
        input_ = input_video_sequence
        for i in range(num_frames):
            output[i, :, :, :] = self.predict_next_frame(input_)
            input_ = np.concatenate((input_video_sequence[i+1:, :, :, :],
                                    output[max(0, i + 1 - input_sequence_length):i+1, :, :, :]), axis=0)
        if front_pad == 0:
           return output
        return np.concatenate((np.zeros([front_pad] + self.frame_shape), output), axis=0)



    def save(self, checkpoint_dir, step, tag=None):
        model_name = "PGN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        if tag is not None:
           model_dir = "%s_%s" % (model_dir, tag)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir=None, tag=None):
        print(" [*] Reading checkpoints...")

        if checkpoint_dir is None:
           if self.checkpoint_dir is None:
              return False
           checkpoint_dir = self.checkpoint_dir

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        if tag is not None:
           model_dir = "%s_%s" % (model_dir, tag)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False









