# Benchmark Pylearn2 tutorial, ready for EC2

Modified from http://www.kurtsp.com/deep-learning-in-python-with-pylearn2-and-amazon-ec2.html


## Setup Vagrant VM

* `brew install caskroom/cask/brew-cask`
* `brew cask install virtualbox vagrant`
* `mkdir test-cnn-box`
* `vagrant init`
* `vagrant up`
 

## Run benchmark
* `vagrant ssh`
* `cd /home/vagrant`
* `cp /vagrant/work.py`
* `time python work.py`
