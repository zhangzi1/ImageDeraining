import os

os.popen("cd ~/tensorflow").read()
# os.popen("source ./bin/activate").read()
os.popen("tensorboard --logdir=~/Desktop/Rain\ Removal/ImageDeraining/graphs --port 6006 --host 0.0.0.0").read()
