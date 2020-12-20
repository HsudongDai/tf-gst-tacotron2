#!python3

import os,sys
import numpy as np
import tensorflow as tf

old_checkpoint_path = sys.argv[1]

style_tokens = tf.contrib.framework.load_variable(old_checkpoint_path, "model/inference/style_tokens")
np.save("style_tokens.npy", style_tokens)

