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
            X_train,
            X_val,
            mlp):

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,LR):
        loss = 0
        penalty = 0
        batches = len(X)/batch_size
        
        output = []

        for i in range(batches):
            new_loss, new_penalty, new_output = train_fn(X[i * batch_size:(i + 1) * batch_size], LR)
            loss += new_loss*batch_size
            penalty += new_penalty*batch_size
            output.append(new_output)
        
        # patch the final partial minibatch
        if len(X) > batches*batch_size:
            new_loss, new_penalty, new_output = train_fn(X[batches*batch_size:len(X)], LR)
            loss += new_loss * (len(X) - batches*batch_size)
            penalty += new_penalty * (len(X) - batches*batch_size)
            output.append(new_output)
        
        loss /= len(X)
        penalty /= len(X)
        
        # in BAE, output is the reconstruction of input X
        # compare output with X, compute the "confuse matrix"
        output = np.concatenate(output)
        # output_bak = output
        output = output > 0
        output = np.int8(output)  # bool to int range -1 0 1 2
        output *= 2
        output -= 1  # BTU
        s00 = np.logical_and(X == -1, output == -1)
        s01 = np.logical_and(X == -1, output == 1)
        s10 = np.logical_and(X == 1, output == -1)
        s11 = np.logical_and(X == 1, output == 1)
        # s = s00 * 1 + s01 * 1 + s10 * 1 + s11 * 1
        # assert (np.all(s == 1))
        s00 = np.sum(s00)
        s01 = np.sum(s01)
        s10 = np.sum(s10)
        s11 = np.sum(s11)
        s = s00 + s01 + s10 + s11
        assert (s == X.shape[0] * X.shape[1])
        print(float(s00) / s, float(s01) / s)
        print(float(s10) / s, float(s11) / s)
        # output = output_bak
        # del output_bak

        # compute the "confuse matrix", considering the group
        # N = output.shape[0]
        # D = output.shape[1]
        # output = output.reshape([N, D / 10, 10])
        # maxIdx = np.uint8(output.argmax(axis=2))  # int range 0-9
        # del output
        # maxIdx = maxIdx.flatten()
        # maxIdx = np.arange(0, maxIdx.shape[0]) * 10 + maxIdx  # int range large
        # output = np.ones([N * D], np.int8)  # int range -1 1
        # output *= -1
        # output[maxIdx] = 1
        # output = output.reshape([N, D])  # max/10
        # s00 = np.logical_and(X == -1, output == -1)
        # s01 = np.logical_and(X == -1, output == 1)
        # s10 = np.logical_and(X == 1, output == -1)
        # s11 = np.logical_and(X == 1, output == 1)
        # # s = s00 * 1 + s01 * 1 + s10 * 1 + s11 * 1
        # # assert (np.all(s == 1))
        # s00 = np.sum(s00)
        # s01 = np.sum(s01)
        # s10 = np.sum(s10)
        # s11 = np.sum(s11)
        # s = s00 + s01 + s10 + s11
        # assert (s == X.shape[0] * X.shape[1])
        # print(float(s00) / s, float(s01) / s)
        # print(float(s10) / s, float(s11) / s)

        return loss, penalty
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X):
        loss = 0
        penalty = 0
        batches = len(X)/batch_size

        output = []

        for i in range(batches):
            new_loss, new_penalty, new_output = val_fn(X[i*batch_size:(i+1)*batch_size])
            loss += new_loss*batch_size
            penalty += new_penalty*batch_size
            output.append(new_output)

        # patch the final partial minibatch
        if len(X) > batches*batch_size:
            new_loss, new_penalty, new_output = val_fn(X[batches*batch_size:len(X)])
            loss += new_loss * (len(X) - batches*batch_size)
            penalty +=  new_penalty * (len(X) - batches*batch_size)
            output.append(new_output)
            
        loss /= len(X)
        penalty /= len(X)

        # in BAE, output is the reconstruction of input X
        # compare output with X, compute the "confuse matrix"
        output = np.concatenate(output)
        # output_bak = output
        output = output > 0
        output = np.int8(output)  # bool to int range -1 0 1 2
        output *= 2
        output -= 1  # BTU
        s00 = np.logical_and(X == -1, output == -1)
        s01 = np.logical_and(X == -1, output == 1)
        s10 = np.logical_and(X == 1, output == -1)
        s11 = np.logical_and(X == 1, output == 1)
        # s = s00 * 1 + s01 * 1 + s10 * 1 + s11 * 1
        # assert (np.all(s == 1))
        s00 = np.sum(s00)
        s01 = np.sum(s01)
        s10 = np.sum(s10)
        s11 = np.sum(s11)
        s = s00 + s01 + s10 + s11
        assert (s == X.shape[0] * X.shape[1])
        print(float(s00) / s, float(s01) / s)
        print(float(s10) / s, float(s11) / s)
        # output = output_bak
        # del output_bak

        # compute the "confuse matrix", considering the group
        # N = output.shape[0]
        # D = output.shape[1]
        # output = output.reshape([N, D / 10, 10])
        # maxIdx = np.uint8(output.argmax(axis=2))  # int range 0-9
        # del output
        # maxIdx = maxIdx.flatten()
        # maxIdx = np.arange(0, maxIdx.shape[0]) * 10 + maxIdx  # int range large
        # output = np.ones([N * D], np.int8)  # int range -1 1
        # output *= -1
        # output[maxIdx] = 1
        # output = output.reshape([N, D])  # max/10
        # s00 = np.logical_and(X == -1, output == -1)
        # s01 = np.logical_and(X == -1, output == 1)
        # s10 = np.logical_and(X == 1, output == -1)
        # s11 = np.logical_and(X == 1, output == 1)
        # # s = s00 * 1 + s01 * 1 + s10 * 1 + s11 * 1
        # # assert (np.all(s == 1))
        # s00 = np.sum(s00)
        # s01 = np.sum(s01)
        # s10 = np.sum(s10)
        # s11 = np.sum(s11)
        # s = s00 + s01 + s10 + s11
        # assert (s == X.shape[0] * X.shape[1])
        # print(float(s00) / s, float(s01) / s)
        # print(float(s10) / s, float(s11) / s)

        return loss, penalty

    # move the big data to child function, clear the big data in the parent function
    # reduce the memory cost of shuffling
    X_train = X_train.move()

    # shuffle the train set
    np.random.shuffle(X_train)
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()

        # do train
        train_loss, train_penalty = train_epoch(X_train, LR)
        np.random.shuffle(X_train)

        # do val
        val_loss, val_penalty = val_epoch(X_val)

        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  training penalty:              "+str(train_penalty))
        print("  validation penalty:            "+str(val_penalty))

        # decay the LR
        LR *= LR_decay

        ### SAVE
        if epoch+1 in (3, 5, 1000, 1168, 2000, 3000, 4000, 5000):
            np.savez("./W-%d.npz"%(epoch+1), *lasagne.layers.get_all_param_values(mlp))# W b BN BN BN BN W b BN BN BN BN


class MoveParameter:

    def __init__(self, obj):
        self.obj = obj

    def move(self):
        r = self.obj
        self.obj = None
        return r
