from __future__ import division
from pylearn2.train import Train
from pylearn2.datasets.mnist import MNIST
from pylearn2.models import softmax_regression, mlp
from pylearn2.training_algorithms import bgd
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions import best_params
from pylearn2.utils import serial
from theano import function
from theano import tensor as T
import numpy as np
import os

h0 = mlp.Sigmoid(layer_name='h0', dim=500, sparse_init=15)
ylayer = mlp.Softmax(layer_name='y', n_classes=10, irange=0)
layers = [h0, ylayer]

model = mlp.MLP(layers, nvis=784)
train = MNIST('train', one_hot=1, start=0, stop=50000)
valid = MNIST('train', one_hot=1, start=50000, stop=60000)
test = MNIST('test', one_hot=1, start=0, stop=10000)

monitoring = dict(valid=valid)
termination = MonitorBased(channel_name="valid_y_misclass")
extensions = [best_params.MonitorBasedSaveBest(channel_name="valid_y_misclass", 
    save_path="train_best.pkl")]
algorithm = bgd.BGD(batch_size=10000, line_search_mode = 'exhaustive', conjugate = 1,
        monitoring_dataset = monitoring, termination_criterion = termination)

save_path = "train_best.pkl"
if os.path.exists(save_path):
    model = serial.load(save_path)
else:
    print 'Running training'
    train_job = Train(train, model, algorithm, extensions=extensions, save_path="train.pkl", save_freq=1)
    train_job.main_loop()

X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)

y = T.argmax(Y, axis=1)
f = function([X], y)
yhat = f(test.X)

y = np.where(test.get_targets())[1]

print 'accuracy', (y==yhat).sum() / y.size