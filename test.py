
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

import pickle
import numpy as np
import matplotlib.pyplot as plt
from cvae import Model, loss_function, split_input_target

print(tf.__version__)

path_to_file = tf.keras.utils.get_file("shakespeare.txt", 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file).read()
vocab = sorted(set(text))
vocab_size = len(vocab)
embedding_dim = 256
units = 1024
checkpoint_dir = "./training_checkpoints"

model = Model(vocab_size, embedding_dim, units)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1,None]))

num_generate = 1000
start_string = "Q"

with open("./c2i_ic2.p","rb") as fi:
    char2idx, idx2char = pickle.load(fi)

input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

text_generated = []
temperature = 1.0

model.reset_states()
for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    predictions = predictions / temperature
    predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])
print(start_string + "".join(text_generated))
