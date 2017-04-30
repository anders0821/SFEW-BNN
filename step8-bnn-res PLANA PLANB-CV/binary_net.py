import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.
    
def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))
    
# The weights' binarization function, 
# taken directly from the BinaryConnect github repository 
# (which was made available by his authors)
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W
    
    else:
        
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)
        
        # Stochastic BinaryConnect
        if stochastic:
        
            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            # print("H = "+str(self.H))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
            
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    def convolve(self, input, deterministic=False, **kwargs):
        
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This function computes the gradient of the binary weights
def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)     

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train, y_train,
            X_val, y_val,
            mlp):

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        loss = 0
        penalty = 0
        acc = 0
        batches = len(X)/batch_size
        
        output = []

        for i in range(batches):
            new_loss, new_penalty, new_acc, new_output = train_fn(X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size], LR)
            loss += new_loss*batch_size
            penalty += new_penalty*batch_size
            acc += new_acc*batch_size
            output.append(new_output)

        # patch the final partial minibatch
        if len(X) > batches*batch_size:
            new_loss, new_penalty, new_acc, new_output = train_fn(X[batches*batch_size:len(X)], y[batches*batch_size:len(X)], LR)
            loss += new_loss * (len(X) - batches*batch_size)
            penalty += new_penalty * (len(X) - batches*batch_size)
            acc += new_acc * (len(X) - batches*batch_size)
            output.append(new_output)
            
        loss /= len(X)
        penalty /= len(X)
        acc /= len(X)

        return loss, penalty, acc

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        loss = 0
        penalty = 0
        acc = 0
        batches = len(X)/batch_size

        output = []

        for i in range(batches):
            new_loss, new_penalty, new_acc, new_output = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            loss += new_loss*batch_size
            penalty += new_penalty*batch_size
            acc += new_acc*batch_size
            output.append(new_output)

        # patch the final partial minibatch
        if len(X) > batches*batch_size:
            new_loss, new_penalty, new_acc, new_output = val_fn(X[batches*batch_size:len(X)], y[batches*batch_size:len(X)])
            loss += new_loss * (len(X) - batches*batch_size)
            penalty +=  new_penalty * (len(X) - batches*batch_size)
            acc +=  new_acc * (len(X) - batches*batch_size)
            output.append(new_output)

        loss /= len(X)
        penalty /= len(X)
        acc /= len(X)

        # fuse
        output = np.concatenate(output)
        output = output.reshape((output.shape[0] / 2, 2, output.shape[1]))
        output = output.mean(1)
        y = y.reshape((y.shape[0] / 2, 2, y.shape[1]))
        y = y.mean(1)
        fuseacc = (output.argmax(1) == y.argmax(1)).mean()

        return loss, penalty, acc, fuseacc

    # move the big data to child function, clear the big data in the parent function
    # reduce the memory cost of shuffling
    X_train = X_train.move()

    # shuffle the train set
    rnd_idx = np.random.permutation(X_train.shape[0])
    X_train = X_train[rnd_idx, :]
    y_train = y_train[rnd_idx, :]
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()

        # do train
        train_loss, train_penalty, train_acc = train_epoch(X_train, y_train, LR)
        rnd_idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[rnd_idx, :]
        y_train = y_train[rnd_idx, :]

        # do val
        val_loss, val_penalty, val_acc, val_fuseacc = val_epoch(X_val, y_val)

        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  training penalty:              "+str(train_penalty))
        print("  validation penalty:            "+str(val_penalty))
        print("  training acc:                  "+str(train_acc))
        print("  validation acc:                "+str(val_acc))
        print("  validation fuseacc:            "+str(val_fuseacc))

        # decay the LR
        LR *= LR_decay

        ### SAVE
        #if epoch+1 in (3, 29532, 29613):
        #    np.savez("./W-%d.npz"%(epoch+1), *lasagne.layers.get_all_param_values(mlp))# W b BN BN BN BN W b BN BN BN BN


class MoveParameter:

    def __init__(self, obj):
        self.obj = obj

    def move(self):
        r = self.obj
        self.obj = None
        return r
