import torch
import sys
sys.path.append('..')

from utils.sdfs import *
from .encoders.encoder import Encoder as Encoder
from .decoders.decoder_sweep import Decoder as Decoder
from .generaters.generater_sweep import Generator

if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()
    generator = Generator()
