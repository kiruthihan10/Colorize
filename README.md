# Colorize
Color the black and white picture with tensorflow

I used Tensorflow to create a deep neaural network with Convelution Neaural Network, Recurrent Neaural Network. 

I also Attached the trained meta graph in this respitory.

# Used Libraries

1. Tensorflow
2. PIL
3. math
4. time
5. numpy
6. glob

# Train
To train the graph You can use color Images with resolution of 2304*1728.

To convert the color images to gray scale you can use the GSCALE_CONVERTER.py.

I used divider parameter to shorten the image data to fit into the Memory.

To create test and train set, I've used test ratio parameter, which divide the numpy array into a test and train

I used Nearon dropout to increase the performance. The dropout rate I used is 0.5. You can change it as you wish.
 
 I used max normalization regularizer to avoid gradient exploding. In this case it is so important to stop the output exceeding above 256.
 
 Kernal regularizer I used here is leaky relu with leak parameter of 0.5
 
 I used 3 convelution layers and a RNN and 3 Deconvelution Layers.
 
 CNN    [4,3,[2,2]]
 CNN    [8,3,[2,2]]
 CNN    [8,3,[2,2]]
 RNN
 DCNN   [9,3,[2,2]]
 DCNN   [3,3,[2,2]]
 DCNN   [3,3,[2,2]]
 
 As this is going to be an array output I used mean squared error as a loss function.
 
 I also used learning rate scheduling to faster converge.
 
 For training time I Used time instead of epochs.
 
 I trained it for 2.5 hours, and got a poor output but with patters extracted. But if You can train it more time, It will converge it to the proper output.
 You can train howlong you want. Because the meta graph only saves when the validation loss is less than the previous min loss.
 
 Finally I also attached a part of code to output a color image of the first image in the folder.
 
 
