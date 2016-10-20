from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.io as io
import numpy as np
import tensorflow as tf 
import numpy as np
from random import shuffle
from sklearn.cross_validation import train_test_split
# from utilities import *
import random
import math
import time

# Command Line Arguments

flags = tf.app.flags

flags.DEFINE_integer('pixels', 11, 'Dimension of input patches')
flags.DEFINE_integer('kernel_dim', 3, 'Dimension of the convolution kernel')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 4000, 'Number of epochs to train.')
flags.DEFINE_integer('fc1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('fc2', 150, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('fc3', 84, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('batch_size', 100, 'Mini Batch size.')
flags.DEFINE_float('dropout', 1.0, ' Amount of droupout for regularization')

opt = flags.FLAGS


# defining global variables

TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
CLASSES = [] 
COUNT = 200 #Number of patches of each class
OUTPUT_CLASSES = 16
NUM_CLASS = 16
TEST_FRAC = 0.25 #Fraction of data to be used for testing

# Data Loading

input_image = io.loadmat('../data/Indian_pines.mat')['indian_pines']
target_image = io.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

input_data = []
targets = []
for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
        if target_image[i,j] != 0:
            input_data.append(input_image[i,j,:])
            targets.append(target_image[i,j])

input_data = np.array(input_data)

CLASSES = {}
for j in range(1, 16):
    temp = []
    for i, target in enumerate(targets):
		if target == j:
			temp.append(input_data[i])
    CLASSES[j] = temp



# for c in range(1, 16):

# 	X = CLASSES[c]
# 	y = [c]*(len(X))
train_data, test_data, train_targets, test_targets = train_test_split(input_data, targets, test_size=0.25, random_state=42)
	# if n == 0:
	# 	TRAIN_PATCH = [X_train]
	# 	TEST_PATCH = [X_test]
	# 	TRAIN_LABELS = [y_train]
	# 	TEST_LABELS = [y_test]
	# else:
	# 	TRAIN_PATCH += X_train
	# 	TEST_PATCH += X_test
	# 	TRAIN_LABELS += y_train
	# 	TEST_LABELS += y_test

# Oversampling

def convert_array(f):
	return np.array(f)

train_data = convert_array(train_data)
test_data = convert_array(test_data)
train_targets = convert_array(train_targets)
test_targets = convert_array(test_targets)


unq, unq_idx = np.unique(train_targets, return_inverse=True)
unq_cnt = np.bincount(unq_idx)
cnt = np.max(unq_cnt)
n_targets = np.empty((cnt*len(unq),) + train_targets.shape[1:], train_targets.dtype)
n_input_patches = np.empty((cnt*len(unq),) + train_data.shape[1:], train_data.dtype)
for j in xrange(len(unq)):
    indices = np.random.choice(np.where(unq_idx==j)[0], cnt)
    n_targets[j*cnt:(j+1)*cnt] = train_targets[indices]
    n_input_patches[j*cnt:(j+1)*cnt] = train_data[indices]
    

x_train = np.asarray(n_input_patches)
x_test = np.asarray(test_data)
y_train = np.asarray(n_targets)
y_test = np.asarray(test_targets)

# placeholders

x = tf.placeholder(tf.float32, [None, 220])
y = tf.placeholder(tf.float32, [None, 16])

def init_placeholders(batch_size):

	"""
	Defining placeholders for data flow

	Args:
		batch_size: mini-batch size
	Returns:
		x, y: placeholder for x and y

	"""

	x = tf.placeholder(tf.float32, [opt.batch_size, IMAGE_PIXELS])
	y = tf.placeholder(tf.float32, [opt.batch_size, NUM_CLASS])

	return x, y

def dense_to_one_hot(labels, n_classes=2):

    """
    Convert class labels from scalars to one-hot vectors.

	Args:
		labels: labels to be encoded
		n_classes: number of classes in the output
	Returns:
		labels_one_hot: encoded labels in OHE

    """
    labels_one_hot = []
    for label in list(labels):
    	temp = np.zeros(n_classes, dtype=np.float32)
    	temp[int(label)-1] = 1
    	labels_one_hot.append(temp)
    labels_one_hot = np.array(labels_one_hot)

    return labels_one_hot

def weight_vector(shape):
	"""
	Generates a weight vector for corresponding shape 

	Args:
		shape: shape of the weight vector
	Returns:
		weight: weight vector

	"""

	weight = tf.Variable(tf.random_normal(shape, mean = 0.0, stddev = 0.01))

	return weight

def bias_vector(shape):
	"""
	Generates bias vector for corresponding shape 

	Args:
		shape: shape of the bias vector
	Returns: 
		bias: bias vector

	"""
	bias = tf.Variable(tf.random_normal(shape, mean = 0.0, stddev = 0.01))

	return bias


def fullyconnected_layer(x, n_units, 
						name = 'fc',
						stddev = 0.02):

	"""
	Generates the fully connected layers with the parameters specified

	Args:
		x: input_vector
		n_units: number of units to connect to 
		name: name of the fully connected layer
		stddev: standard deviation for variable initialization
	Returns:
		fc: fully connected outputs

	"""
	shape = x.get_shape().as_list()

	with tf.variable_scope(name) as scope:
		weights = tf.Variable(
		    tf.truncated_normal([shape[1], n_units],
		                        stddev=1.0 / math.sqrt(float(shape[1]))),
		    name='weights')
		biases = tf.Variable(tf.zeros([n_units]),
		                     name='biases')
		fc = tf.nn.relu(tf.matmul(x, weights) + biases, name=name)

		return fc



def softmax(x, n_units, name='softmax'):
	"""
	Softmax layer on top of the fully connected layer

	Args:
		x: input vector
		n_units: number of units to connect to 
	Returns:
		logits: logit vector containing scores for every class

	"""
	with tf.variable_scope(name) as scope:
		weights = tf.Variable(
			tf.truncated_normal([n_units, NUM_CLASS],
										stddev=1.0 / math.sqrt(float(n_units))),
			name='weights')
		biases = tf.Variable(tf.zeros([NUM_CLASS]),
							name='biases')

		logits = tf.matmul(x, weights) + biases

		return logits



def eval_loss(logits, labels):
	"""
	Calculates the loss from the logits and the labels.

	Args:
		logits: Logits tensor, float - [batch_size, NUM_CLASSES].
		labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].
	Returns:
		loss: Loss tensor of type float.

	"""
	# labels = tf.to_float32(labels)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
												logits, labels, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	return loss

def optimizer_step(loss, learning_rate):
	"""
	Create summary to keep track of the loss after iterations in 
	tensorboard and optimizes the loss function using Adam Optimizer

	Args:
		loss: loss value obtained from eval_loss()
		learning_rate: learning rate for SGD
	Returns:
		optim_op: optimizer op

	"""
	# adding summary to keep track of loss
	tf.scalar_summary(loss.op.name, loss)
	# using Adam Optimizer as the optimizer function
	optimizer = tf.train.AdamOptimizer(learning_rate)
	# track the global step
	global_step = tf.Variable(0, name='global_step', trainable=False)
	# use the optimizer to minimize the loss 
	optim_op = optimizer.minimize(loss, global_step=global_step)

	return optim_op

def evaluate(logits, labels):
	"""
	Evaluating accuracy of the network

	Args:
		logits: logits tensor
		labels: labels corresponding to a particular batch
	Returns:
		accuracy: accuracy of the network

	"""
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
	return accuracy

def evaluate_model(sess, eval_correct, x, y, x_data, y_data, opt):
	"""
	Evaluates the model 
	Args:
		sess: working session on graph
		x: placeholder for patch
		y: placeholder for label
		x_data: input patch
		y_data: output label
	Returns:

	"""
	true_count = 0  # Counts the number of correct predictions.
	num_batch = int(len(x_data)/opt.batch_size)
	num_examples = num_batch*opt.batch_size
	for i in range(num_batch + 1):

		if i == num_batch:
			batch_xs = x_data[i*opt.batch_size:]
			batch_ys = y_data[i*opt.batch_size:]
		else:
			batch_xs = x_data[i*opt.batch_size:(i+1)*opt.batch_size]
			batch_ys = y_data[i*opt.batch_size:(i+1)*opt.batch_size]

		# batch_xs = batch_xs.reshape(opt.batch_size, 220)

		true_count += sess.run(eval_correct,
		                       feed_dict={
		                    		x: batch_xs,
		                    		y: batch_ys,
		                       })
	precision = true_count / num_examples
	print(' Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


# with tf.Graph().as_default():
y_test = dense_to_one_hot(y_test, 16)
y_train = dense_to_one_hot(y_train, 16)
# fc1 
W_fc1 = weight_vector([220, opt.fc1])
b_fc1 = bias_vector([opt.fc1])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

# fc2
W_fc2 = weight_vector([opt.fc1, opt.fc2])
b_fc2 = bias_vector([opt.fc2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# fc3
W_fc3 = weight_vector([opt.fc2, opt.fc3])
b_fc3 = bias_vector([opt.fc3])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

# dropout
h_fc3_drop = tf.nn.dropout(h_fc3, opt.dropout)

# softmax
logits = softmax(h_fc3_drop, opt.fc3, name="softmax")

# sess = tf.session()

loss = eval_loss(logits, y)

optim_op = optimizer_step(loss, opt.learning_rate)
# correct eval
eval_correct = evaluate(logits, y)

# initializing all variables on the graph
init = tf.initialize_all_variables()

# saver for recording training 
saver = tf.train.Saver()

# creating a session for running ops on graph
sess = tf.Session()

# run initializer in the defined session
sess.run(init)

num_batch = int(len(x_train)/opt.batch_size)

for epoch in xrange(opt.num_epochs):
	# start time
	start = time.time()

	for i in range(num_batch + 1):

		if i == num_batch:
			batch_xs = x_train[i*opt.batch_size:]
			batch_ys = y_train[i*opt.batch_size:]
		else:
			batch_xs = x_train[i*opt.batch_size:(i+1)*opt.batch_size]
			batch_ys = y_train[i*opt.batch_size:(i+1)*opt.batch_size]

		# batch_xs = batch_xs.reshape(opt.batch_size, 220)

		_, loss_value = sess.run([optim_op, loss],
		                        feed_dict={
		                    		x: batch_xs,
		                    		y: batch_ys
		                        })

	end_time = time.time() - start

	if epoch % 2 == 0:
		print('Epoch %d: loss = %.2f (%.3f sec)' % (epoch, loss_value, end_time))

	if (epoch + 1) % 10 == 0 or (epoch + 1) == opt.num_epochs:
		# saver.save(sess, 'model-STN-CNN-'+str(opt.pixels)+'X'+str(opt.pixels)+'.ckpt', global_step=step)
		# Evaluate on train data
		print("Training Data Evaluation:")
		evaluate_model(sess, eval_correct, x, y, x_train, y_train, opt)

		print(" Test Data Evaluation:")
		evaluate_model(sess, eval_correct, x, y, x_test, y_test, opt)


