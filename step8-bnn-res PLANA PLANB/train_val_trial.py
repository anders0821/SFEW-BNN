from __future__ import print_function

import sys
import os
import time

import numpy as np

# for reproducibility
# different cpu platform can reproduce same result
# cpu / gpu reproduce can different result
np.random.seed(1234)

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import lasagne # lasagne must be imported before theano

import theano
import theano.tensor as T

import cPickle as pickle
import gzip

import binary_net

from collections import OrderedDict

import math
import SFEW2

def trial(N_HIDDEN_LAYERS, NUM_UNITS, OUTPUT_TYPE, MAIN_LOSS_TYPE, LAMBDA, FOLD, FINTUNE_SNAPSHOT, FINTUNE_SCALE):
    # BN parameters
    batch_size = 97
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    # alpha = .15
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))

    # MLP parameters
    #NUM_UNITS = 25
    print("NUM_UNITS = "+str(NUM_UNITS))
    #N_HIDDEN_LAYERS = 1
    print("N_HIDDEN_LAYERS = "+str(N_HIDDEN_LAYERS))

    # Training parameters
    num_epochs = 1000000
    print("num_epochs = "+str(num_epochs))

    # Dropout parameters
    dropout_in = .2 # 0. means no dropout
    print("dropout_in = "+str(dropout_in))
    dropout_hidden = .5
    print("dropout_hidden = "+str(dropout_hidden))

    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")

    # BinaryConnect
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))

    # Decaying LR
    #LR_start = .003
    LR_start = 0.000003
    print("LR_start = "+str(LR_start))
    #LR_fin = 0.0000003
    LR_fin = LR_start
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    # replace the dataset
    print('Loading SFEW2 dataset...')
    [train_x, train_y, val_x, val_y] = SFEW2.load_train_val()
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    print('last training minibatch size: '+str(train_x.shape[0]-train_x.shape[0]/batch_size*batch_size)+' / '+str(batch_size))
    print('last training minibatch size should not be too small (except 0). try decrease the batch_size, but not add more minibatches.')
    print('minibatches size: '+str(batch_size))
    print('suggested minibatches size: '+str(math.ceil(float(train_x.shape[0])/math.ceil(float(train_x.shape[0])/100))))

    print('Building the MLP...')
    # Prepare Theano variables for inputs and targets
    input = T.matrix('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    mlp = lasagne.layers.InputLayer(
            shape=(None, train_x.shape[1]),
            input_var=input)
            
    mlp = lasagne.layers.DropoutLayer(
            mlp, 
            p=dropout_in)

    '''
    hidden layers
    k=0 MDLBP-1500 with pretrain-finetune
    k=1 1500-100
    k=2 100-100 with res shutcut
    '''
    for k in range(N_HIDDEN_LAYERS):
        # res shutcut
        if(k>=1):
            mlp_res = mlp
            print('backup mlp_res')

        # pretrain-finetune
        if(k==0):
            # fixed num_units
            mlp = binary_net.DenseLayer(
                    mlp,
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=1500)
            # scale down the LR of transfered dense layer
            print('scale down the LR of transfered dense layer from', str(mlp.W_LR_scale))
            mlp.W_LR_scale *= np.float32(FINTUNE_SCALE)
            print('to', str(mlp.W_LR_scale))
        else:
            mlp = binary_net.DenseLayer(
                mlp,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=NUM_UNITS)

        mlp = lasagne.layers.BatchNormLayer(
                mlp,
                epsilon=epsilon,
                alpha=alpha)

        mlp = lasagne.layers.NonlinearityLayer(
                mlp,
                nonlinearity=activation)

        # res shutcut
        if(k>=1):
            # mlp = lasagne.layers.ElemwiseSumLayer([mlp, mlp_res]) # + -> {-1 +1} {-2 -1 0 +1 +2}
            mlp = lasagne.layers.ElemwiseMergeLayer([mlp, mlp_res], T.maximum) # maximum -> bitwise or
            del mlp_res
            print('or mlp_res')

        mlp = lasagne.layers.DropoutLayer(
                mlp, 
                p=dropout_hidden)

        # pretrain-finetune
        # only restore the first layer group
        if (k == 0):
            if (FINTUNE_SNAPSHOT != 0):
                print('Load ./W-%d.npz' % FINTUNE_SNAPSHOT)
                with np.load('./W-%d.npz' % FINTUNE_SNAPSHOT) as f:
                    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                param_values = param_values[0:6]
                lasagne.layers.set_all_param_values(mlp, param_values)

    mlp = binary_net.DenseLayer(
                mlp, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=7)

    mlp = lasagne.layers.BatchNormLayer(
            mlp,
            epsilon=epsilon,
            alpha=alpha)

    # network output BN or SGN
    if OUTPUT_TYPE=='C':
        pass#
    elif OUTPUT_TYPE=='D':
        mlp = lasagne.layers.NonlinearityLayer(mlp, nonlinearity=activation)
    else:
        assert(False)

    # loss weight nodes
    SPARSITY = 0.9
    SPARSITY_MAP = (np.float32(train_x==-1)).mean(0)
    LOSS_WEIGHT_1 = 1.+input*(2.*SPARSITY-1)
    LOSS_WEIGHT_1 /= 4*SPARSITY*(1 - SPARSITY)# fixed 1->-1:5 -1->1:5/9 weights
    LOSS_WEIGHT_2 = 1.+input*(2.*SPARSITY_MAP-1)#
    LOSS_WEIGHT_2 /= 4*SPARSITY_MAP*(1 - SPARSITY_MAP)# weights considering element's prior probability

    # train loss nodes
    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    if MAIN_LOSS_TYPE=='SH':
        train_loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    elif MAIN_LOSS_TYPE == 'W1SH':
        train_loss = T.mean(T.sqr(T.maximum(0., (1. - target * train_output))) * LOSS_WEIGHT_1)
    elif MAIN_LOSS_TYPE == 'W2SH':
        train_loss = T.mean(T.sqr(T.maximum(0., (1. - target * train_output))) * LOSS_WEIGHT_2)
    elif MAIN_LOSS_TYPE == 'H':
        train_loss = T.mean(T.maximum(0.,1.-target*train_output))
    elif MAIN_LOSS_TYPE == 'W1H':
        train_loss = T.mean(T.maximum(0., (1. - target * train_output)) * LOSS_WEIGHT_1)
    elif MAIN_LOSS_TYPE == 'W2H':
        train_loss = T.mean(T.maximum(0., (1. - target * train_output)) * LOSS_WEIGHT_2)
    else:
        assert(False)

    # + sparse penalty
    if LAMBDA>0:
        train_pixel_wise_density = T.mean(T.reshape((train_output+1.)/2., [train_output.shape[0], train_output.shape[1]/10, 10]), axis=2)
        train_penalty = LAMBDA*T.mean(T.sqr(train_pixel_wise_density - (1.-SPARSITY)))
    else:
        train_penalty = T.constant(0.)
    train_loss = train_loss + train_penalty

    # acc
    train_acc = T.mean(T.eq(T.argmax(train_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # grad nodes
    if binary:
        # W updates
        W = lasagne.layers.get_all_params(mlp, binary=True)
        W_grads = binary_net.compute_grads(train_loss,mlp)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,mlp)

        # other parameters updates
        params = lasagne.layers.get_all_params(mlp, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=train_loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(mlp, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=train_loss, params=params, learning_rate=LR)

    # val loss nodes
    # must be created after grad nodes
    val_output = lasagne.layers.get_output(mlp, deterministic=True)
    if MAIN_LOSS_TYPE=='SH':
        val_loss = T.mean(T.sqr(T.maximum(0.,1.-target*val_output)))
    elif MAIN_LOSS_TYPE == 'W1SH':
        val_loss = T.mean(T.sqr(T.maximum(0., (1. - target * val_output))) * LOSS_WEIGHT_1)
    elif MAIN_LOSS_TYPE == 'W2SH':
        val_loss = T.mean(T.sqr(T.maximum(0., (1. - target * val_output))) * LOSS_WEIGHT_2)
    elif MAIN_LOSS_TYPE == 'H':
        val_loss = T.mean(T.maximum(0.,1.-target*val_output))
    elif MAIN_LOSS_TYPE == 'W1H':
        val_loss = T.mean(T.maximum(0., (1. - target * val_output)) * LOSS_WEIGHT_1)
    elif MAIN_LOSS_TYPE == 'W2H':
        val_loss = T.mean(T.maximum(0., (1. - target * val_output)) * LOSS_WEIGHT_2)

    # + sparse penalty
    if LAMBDA>0:
        val_pixel_wise_density = T.mean(T.reshape((val_output + 1.) / 2., [val_output.shape[0], val_output.shape[1] / 10, 10]), axis=2)
        val_penalty = LAMBDA*T.mean(T.sqr(val_pixel_wise_density - (1. - SPARSITY)))
    else:
        val_penalty = T.constant(0.)
    val_loss = val_loss + val_penalty

    # acc
    val_acc = T.mean(T.eq(T.argmax(val_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
    # and returning the corresponding training train_loss:
    train_fn = theano.function([input, target, LR], [train_loss, train_penalty, train_acc, train_output], updates=updates)

    # Compile a second function computing the validation train_loss and accuracy:
    val_fn = theano.function([input, target], [val_loss, val_penalty, val_acc, val_output])

    print('Training...')
    train_x = binary_net.MoveParameter(train_x)
    binary_net.train(train_fn,val_fn,
        batch_size,
        LR_start,LR_decay,
        num_epochs,
        train_x, train_y,
        val_x, val_y,
        mlp)


if __name__ == "__main__":
    assert(len(sys.argv)==9)
    N_HIDDEN_LAYERS = int(sys.argv[1])
    NUM_UNITS = int(sys.argv[2])
    OUTPUT_TYPE = sys.argv[3]  # C:continuous | D:discrete
    MAIN_LOSS_TYPE = sys.argv[4]  # SH:squared hinge | W1SH:weighted1 squared hinge  | W2SH:weighted2 squared hinge | H:hinge | W1H:weighted1 hinge | W2H:weighted2 hinge
    LAMBDA = float(sys.argv[5])
    FOLD = int(sys.argv[6])
    FINTUNE_SNAPSHOT = int(sys.argv[7])
    FINTUNE_SCALE = float(sys.argv[8])

    assert 1 <= N_HIDDEN_LAYERS
    assert 10 <= NUM_UNITS
    assert LAMBDA <= 0.
    assert 1 <= FOLD <= 5
    
    print(N_HIDDEN_LAYERS, NUM_UNITS, OUTPUT_TYPE, MAIN_LOSS_TYPE, LAMBDA, FOLD, FINTUNE_SNAPSHOT, FINTUNE_SCALE)
    print('TRIAL BEGIN')
    trial(N_HIDDEN_LAYERS, NUM_UNITS, OUTPUT_TYPE, MAIN_LOSS_TYPE, LAMBDA, FOLD, FINTUNE_SNAPSHOT, FINTUNE_SCALE)
    print('TRIAL END')
