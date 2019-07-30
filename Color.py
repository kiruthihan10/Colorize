from PIL import Image
import glob
import numpy
import tensorflow as tf
from tensorflow import contrib
from math import exp
import time
from math import sqrt

#########################################################################################################

var_X = []
var_Y = []

divider = 3*3#*2*2

for filename in glob.glob('C:/Users/kirut/Documents/Project color/Target/*.jpg'):
    print("Opening "+filename)
    i = Image.open(filename)
    i = i.resize((int(2304/divider),int(1728/divider)),Image.ANTIALIAS)
    mat_i = numpy.asarray(i)
    var_Y.append(mat_i)



for filename in glob.glob('C:/Users/kirut/Documents/Project color/Training/*.png'):
    print("Opening "+filename)
    i = Image.open(filename)
    i = i.resize((int(2304/divider),int(1728/divider)),Image.ANTIALIAS)
    mat_i = numpy.asarray(i)
    var_X.append(mat_i)





train_X = var_X[:350]
train_Y = var_Y[:350]
test_X = var_X[350:]
test_Y = var_Y[350:]

print("Images Loaded")

##########################################################################################################

random_i = Image.open('C:/Users/kirut/Documents/Project color/Target/Picture 001.jpg')
height, width = random_i.size

n_outchannels = 3

default_size = 2304*1728/(divider**2)
dropoutrate = 0.5

print("Constants Fixed")

##########################################################################################################

threshold = 100.0

def max_norm_regularizer(threshold, axes=None, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None # there is no regularization loss term
    return max_norm

max_norm_reg = max_norm_regularizer(threshold=threshold)

clip_all_weights = tf.get_collection("max_norm")

##########################################################################################################

def leaky_relu(z,name=None):
    return tf.maximum(0.5*z,z,name=name)

from functools import partial

##def leaky_relu(z,name=None):
##    return tf.nn.elu(z)

he_init = tf.contrib.layers.variance_scaling_initializer()

##########################################################################################################

X = tf.placeholder(shape=(None,width/divider,height/divider,2),dtype=tf.float32)
print(X)
training = tf.placeholder_with_default(False,shape=(),name='training')

X_drop = tf.layers.dropout(X,dropoutrate)
my_batch_norm_layer = partial(tf.layers.batch_normalization,training=training,momentum=0.9)
bn0 = my_batch_norm_layer(X_drop)
bn0_act = leaky_relu(bn0)
print(bn0_act)

conv1 = tf.layers.conv2d(bn0_act,filters=2**2,kernel_size=3,strides=[2,2],padding="SAME",kernel_regularizer=max_norm_reg,activation=leaky_relu)
print(conv1)
conv1_drop = tf.layers.dropout(conv1,dropoutrate)
##avg_pool1 = tf.nn.avg_pool(conv1_drop,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
##print(avg_pool1)
bn1 = my_batch_norm_layer(conv1_drop)
##bn1_act = leaky_relu(bn1)
print(bn1)

conv2 = tf.layers.conv2d(bn1,filters=2**3,kernel_size=3,strides=[2,2],padding="SAME",kernel_regularizer=max_norm_reg,activation=leaky_relu)
print(conv2)
conv2_drop = tf.layers.dropout(conv2,dropoutrate)
bn2 = my_batch_norm_layer(conv2_drop)
print(bn2)

conv3 = tf.layers.conv2d(bn2,filters=2**3,kernel_size=3,strides=[2,2],padding="SAME",kernel_regularizer=max_norm_reg,activation=leaky_relu)
print(conv3)
conv3_drop = tf.layers.dropout(conv3,dropoutrate)
bn3 = my_batch_norm_layer(conv3_drop)
print(bn3)

flatten = tf.layers.Flatten()(bn3)
reshaped1 = tf.reshape(flatten,[tf.shape(flatten)[0],1,int(flatten.shape[1])])
print(flatten)
print(reshaped1)

lstm0 = tf.nn.rnn_cell.LSTMCell(num_units=int(reshaped1.shape[2]),activation=leaky_relu,use_peepholes=True)
output0,states0 = tf.nn.dynamic_rnn(lstm0,reshaped1,dtype=tf.float32)
print(lstm0)
print(states0)
print(output0)

print(bn2_act.shape)
reshaped2 = tf.reshape(output0,[tf.shape(output0)[0],int(width/(divider*8)),int(height/(divider*8)),8])
print(reshaped2)

deconv1 = tf.layers.conv2d_transpose(reshaped2,filters=3**2,kernel_size=3,strides=[2,2],padding="SAME",activation=leaky_relu,kernel_regularizer=max_norm_reg)
print(deconv1)
deconv1_drop = tf.layers.dropout(deconv1,dropoutrate)
bn3= my_batch_norm_layer(deconv1_drop)

deconv2 = tf.layers.conv2d_transpose(bn3,filters=3**1,kernel_size=3,strides=[2,2],padding="SAME",activation=leaky_relu,kernel_regularizer=max_norm_reg)
print(deconv2)
deconv2_drop = tf.layers.dropout(deconv2,dropoutrate)
bn4= my_batch_norm_layer(deconv2_drop)

deconv3 = tf.layers.conv2d_transpose(bn4,filters=3**1,kernel_size=3,strides=[2,2],padding="SAME",activation=leaky_relu,kernel_regularizer=max_norm_reg)
print(deconv3)
deconv3_drop = tf.layers.dropout(deconv3,dropoutrate)
bn5= my_batch_norm_layer(deconv3_drop)


logits = bn5

print(logits)

y = tf.placeholder(shape=(None,int(width/divider),int(height/divider),3),dtype=tf.float32)
print(y)

print("Layers Defined")

##########################################################################################################

with tf.name_scope("loss"):
    loss = tf.compat.v1.losses.mean_squared_error(labels=y,predictions=logits)

print("Loss function defined")

##########################################################################################################

with tf.name_scope("train"):

    initial_learning_rate = 0.05
    decay_steps = 10000
    decay_rate = 1/10
    global_step = tf.Variable(0,trainable=False,name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,decay_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss,global_step =global_step)

print("train function defined")

##########################################################################################################

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 70
print("Global variables intitalized")

##########################################################################################################

tfe = contrib.eager

now = time.time()

train_time = 2.5*60*60    #in seconds

with tf.Session() as sess:
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    sess.run(tf.local_variables_initializer(),options=run_options)
    init.run()
    print("Session started running")
    epoch = 0
    max_val_error = numpy.inf
    error_count = 0
    while time.time()-now<train_time:
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()
        n=10
        print(epoch)
        for iteration in range(len(train_X)//batch_size):
            X_batch = train_X[:n+batch_size+1]
            Y_batch = train_Y[:n+batch_size+1]
            sess.run(training_op,feed_dict={X:X_batch,y:Y_batch})
            sess.run(clip_all_weights)
            n=n+batch_size+1
            acc_batch = loss.eval(feed_dict={X: X_batch, y: Y_batch})
            acc_valid = loss.eval(feed_dict={X: test_X, y: test_Y})
            print(epoch, "Batch loss:", sqrt(acc_batch/fc2_size), "Validation loss:", sqrt(acc_valid/fc2_size))
            epoch+=1
        if acc_valid < max_val_error:
            max_val_error = acc_valid
            save_path = saver.save(sess, "./my_model_finalmain.ckpt")


##########################################################################################################

for filename in glob.glob('C:/Users/kirut/Documents/Project color/Target/*.jpg'):
    print("Opening "+filename)
    i = Image.open(filename)
    i = i.resize((int(2304/divider),int(1728/divider)),Image.ANTIALIAS)
    mat_i = numpy.asarray(i)
    pixels = mat_i.flatten().reshape(int(int(2304*1728*3)/(divider**2)))
    mat_i = numpy.array(pixels)
    checky=mat_i
    break

for filename in glob.glob('C:/Users/kirut/Documents/Project color/Training/*.png'):
    print("Opening "+filename)
    i = Image.open(filename)
    i = i.resize((int(2304/divider),int(1728/divider)),Image.ANTIALIAS)
    mat_i = numpy.asarray(i)
    checkx = mat_i
    break

##########################################################################################################

with tf.Session() as sess:
    saver.restore(sess, "./my_model_finalmain.ckpt") # or better, use save_path
    X_new_scaled = [checkx]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    print(X)
    print(numpy.array(X_new_scaled).shape)

##########################################################################################################

def toim(lst):
    lst = numpy.array(lst).reshape(int(1728/divider),int(2304/divider),3)
    print(lst.shape)
    return lst

##########################################################################################################

predicted_array = toim(Z).astype('uint8')
print("predicted_array")
print(predicted_array)
print(predicted_array.shape)
im=Image.fromarray(predicted_array)
im.show()
original_array = toim(checky).astype('uint8')
print("originial_array")
print(original_array)
print(original_array.shape)
print("difference")
print(predicted_array-original_array)
im=Image.fromarray(original_array)
mse = ((original_array - predicted_array)**2).mean(axis=None)
print(mse)
im.show()
input()
