import numpy as np

from layers import relu_forward, fc_forward, fc_backward, relu_backward, softmax_loss
from cnn_layers import conv_forward, conv_backward, max_pool_forward, max_pool_backward


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on the ipython notebook.
    """
    print("Hello from cnn.py!")


class ConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - fc - relu - fc - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_filters_1=6, num_filters_2=16, filter_size=5,
               hidden_dim=100, num_classes=10, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters_1: Number of filters to use in the first convolutional layer
        - num_filters_2: Number of filters to use in the second convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        (self.C, self.H, self.W) = input_dim
        self.filter_size = filter_size
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Your initializations should work for any valid input dims,      #
        # number of filters, hidden dims, and num_classes. Assume that we use      #
        # max pooling with pool height and width 2 with stride 2.                  #
        #                                                                          #
        # For Linear layers, weights and biases should be initialized from a       #
        # uniform distribution from -sqrt(k) to sqrt(k),                           #
        # where k = 1 / (#input features)                                          #
        # For Conv. layers, weights should be initialized from a uniform           #
        # distribution from -sqrt(k) to sqrt(k),                                   #
        # where k = 1 / ((#input channels) * filter_size^2)                        #
        # Note: we use the same initialization as pytorch.                         #
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html           #
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html           #
        #                                                                          #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights for the convolutional layer using the keys 'W1' and 'W2'   #
        # (here we do not consider the bias term in the convolutional layer);      #
        # use keys 'W3' and 'b3' for the weights and biases of the                 #
        # hidden fully-connected layer, and keys 'W4' and 'b4' for the weights     #
        # and biases of the output affine layer.                                   #
        #                                                                          #
        # Make sure you have initialized W1, W2, W3, W4, b3, and b4 in the         #
        # params dicitionary.                                                      #
        #                                                                          #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3. Calculate the size of W3 dynamically           #
        ############################################################################
        k1 = 1 / (self.C * filter_size ** 2)
        self.params['W1'] = np.random.uniform(-np.sqrt(k1), np.sqrt(k1), 
                                              (num_filters_1, self.C, filter_size, filter_size)).astype(dtype)
        
        # Second convolutional layer
        k2 = 1 / (num_filters_1 * filter_size ** 2)
        self.params['W2'] = np.random.uniform(-np.sqrt(k2), np.sqrt(k2), 
                                              (num_filters_2, num_filters_1, filter_size, filter_size)).astype(dtype)

        H_pool = (self.H - self.filter_size + 1) // 2  
        H_pool = (H_pool - self.filter_size + 1) // 2  

        W_pool = (self.W - self.filter_size + 1) // 2 
        W_pool = (W_pool - self.filter_size + 1) // 2 

        flattened_dim = num_filters_2 * H_pool * W_pool  

        # Fully connected hidden layer
        k3 = 1 / flattened_dim
        self.params['W3'] = np.random.uniform(-np.sqrt(k3), np.sqrt(k3), 
                                              (flattened_dim, hidden_dim)).astype(dtype)
        self.params['b3'] = np.zeros(hidden_dim, dtype=dtype)

        # Fully connected output layer
        k4 = 1 / hidden_dim
        self.params['W4'] = np.random.uniform(-np.sqrt(k4), np.sqrt(k4), 
                                              (hidden_dim, num_classes)).astype(dtype)
        self.params['b4'] = np.zeros(num_classes, dtype=dtype)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1 = self.params['W1']
        W2 = self.params['W2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        # Hint: The output of max pooling after W2 needs to be flattened before    #
        # it can be passed into W3.                                                #
        ############################################################################
        conv1_out, cache1= conv_forward(X, W1)  
        relu1_out = np.maximum(0, conv1_out)  
        pool1_out, cache_pool1 = max_pool_forward(relu1_out, pool_param)  

        conv2_out, cache2 = conv_forward(pool1_out, W2)  
        relu2_out = np.maximum(0, conv2_out)  
        pool2_out, cache_pool2 = max_pool_forward(relu2_out, pool_param)  

        N, F, H, W = pool2_out.shape
        flatten_out = pool2_out.reshape(N, -1)

        
        hidden_out = np.maximum(0, flatten_out @ W3 + b3)

        scores = hidden_out @ W4 + b4        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k].                                                      #
        # Hint: The backwards from W3 needs to be un-flattened before it can be    #
        # passed into the max pool backwards                                       #
        ############################################################################
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = X.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        grads = {}

        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads['W4'] = hidden_out.T @ dscores
        grads['b4'] = np.sum(dscores, axis=0)

        dhidden = dscores @ W4.T
        dhidden[hidden_out <= 0] = 0 

        grads['W3'] = flatten_out.T @ dhidden 
        grads['b3'] = np.sum(dhidden, axis=0)

        dflatten_out = dhidden @ W3.T
        dpool2_out = dflatten_out.reshape(pool2_out.shape)

        drelu2_out = max_pool_backward(dpool2_out, cache_pool2)
        drelu2_out[relu2_out <= 0] = 0  
        dconv2_out, dW2 = conv_backward(drelu2_out, cache2)

        grads['W2'] = dW2

        dpool1_out = max_pool_backward(dconv2_out, cache_pool1)
        drelu1_out = dpool1_out
        drelu1_out[relu1_out <= 0] = 0
        dconv1_out, dW1 = conv_backward(drelu1_out, cache1)

        grads['W1'] = dW1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
