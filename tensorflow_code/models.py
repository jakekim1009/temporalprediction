#CNN encodes the image into 100-dimensional vectors.
#LSTM-RNN outputs the Nightlight values.

import tensorflow as tf

def LSTM(input_sequence, name, num_hidden_units, reuse=False, train=True):
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
        init_state = cell.zero_state(batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, input_sequence,
                       initial_state=init_state, time_major=True)
        return outputs
        #Outputs are in [max_time, batch_size, cell.output_size = num_hidden_units]
        #How to change the output into ond-dimensional scalar?

    # End simple tensor flow LSTM implementation


#Encodes satellite image video by applying ResNet on each image of the sequence.
def video_encoder(video_sequence):




video_sequence_length = 14
input_sequence_length = 14
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
frame_shape = [IMAGE_SIZE, IMAGE_SIZE]
batch_size = 10

def inference(images, inference_name, resnet_hidden_units, lstm_hidden_units):

    # place the entire video sequence in the placeholder.
    video_sequence = tf.placeholder(tf.float32, [video_sequence_length, batch_size] + frame_shape)

    #Only select the part of the video that corresponds to the input sequence length.
    #input_sequence = tf.slice(video_sequence, [0, 0, 0, 0, 0], [input_sequence_length, -1, -1, -1, -1])

    with tf.variable_scope(inference_name, reuse=reuse) as scope:
        num_frames, num_batches, image_height, image_width, image_channels = video_sequence.get_shape().as_list()
        encoded = encoder(video_sequence, "encoder", reuse=reuse, train=train)
        lstm_output = LSTM(encoded, input_sequence_length, "lstm", num_LSTM_units, reuse=reuse, train=train)
        decoded = decoder(lstm_output, image_height, image_width, image_channels, "decoder", reuse=reuse, train=train)
        return ((decoded + 1) * max_pixel_val) / 2.0, lstm_output, encoded


def loss(labels):
    # The ground truth(label) is the Nightlights intensity which will be fed later.
    ground_truth = tf.placeholder(tf.float32, [video_sequence_length, batch_size])


lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
# Initial state of the LSTM memory.
state = tf.zeros([batch_size, lstm.state_size])

loss = 0.0
for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, target_words)

