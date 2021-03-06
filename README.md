CS 760 Homework Assignment :

neural network using stochastic gradient descent (online training). It accepts three commandline arguments as follows:
nnet l h e <train‐set‐file> <test‐set‐file>
where l specifies the learning rate, h the number of hidden units and e the number of training epochs. After training for e 
epochs on the training set, you should use the learned neural net to predict a classification for every instance in the test set.

The network is intended for binary classification problems, and therefore it has one output unit with a
sigmoid function. The sigmoid should be trained to predict 0 for the first class listed in the given ARFF files, and 1 for the 
second class.
Stochasic gradient descent is used to minimize crossentropy error.
If h = 0, the network should have no hidden units, and the input units should be directly connected to the output unit. 
Otherwise, if h > 0, the network should have a single layer of h hidden units with each fully connected to the input units and 
the output unit. For each numeric feature, one input unit is used. For each discrete feature, a one of k encoding. is used.
To ensure that hidden unit activations don't get saturated, standardization is used for numeric features

Each epoch is one complete pass through the training instances. The order of the training instances is randomized before 
starting training, but each epoch goes through the instances in the same order.
All weights and bias parameters are initialized to random values in [0.01,0.01].

The program handles numeric and nominal attributes, and simple ARFF files.

Output: 
After each training epoch, the epoch number (starting from 1), the crossentropy
error, the number of training instances that are correctly classified, and the number of instances that are are misclassified
are printed on one line separated by tabs. 
To determine a classification, a threshold of 0.5 is used on the activation of the output unit (i.e. the value computed by the 
sigmoid).
After training, for each test instance the activation of the output unit, the predicted class, and the
correct class are printed. These values are tabseparated with one line per test instance.
Finally, the number of correctly classified and the number of incorrectly classified test instances
when a threshold of 0.5 is used on the activation of the output unit are printed.
