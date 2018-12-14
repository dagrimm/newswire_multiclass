# newswire_multiclass
Multiclass classification of Keras newswire dataset using Tensorflow backend. 

Using one-hot encoding for categories. 
Activation of neurons is relu. 
Output is softmax probability with 46 neurons - (45 categories and 0)

Optimizer is rmsprop. 
Loss is computed with categorical crossentropy. 
Metric is accuracy. 

Learning rate is plotted with Matplotlib - of course you could add a callback for TensorBorad. 
