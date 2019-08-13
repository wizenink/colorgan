from train import *
import argparse

parser = argparse.ArgumentParser(description="Insert correct arguments")
parser.add_argument('--input', action="store", dest="input")
parser.add_argument('--fake', action="store", dest="fake")
parser.add_argument('--batch', action="store", dest="batch",type=int)

t = Trainer()
t.pretrain_generator(parser.batch,130,10,1000,parser.input,parser.fake)
t.pretrain_discriminator(parser.batch,100,parser.input,parser.fake)
t.train(parser.batch,100,100,parser.input)

