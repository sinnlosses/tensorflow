
from __future__ import absolute_import, division, print_function

import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

import os
import time
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super(Model, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        if tf.test.is_gpu_available():
            self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                                return_sequences=True,
                                                recurrent_initializer="glorot_uniform",
                                                stateful=True)
        else:
            self.gru = tf.keras.layers.GRU(self.units,
                                            return_sequences=True,
                                            return_activation="sigmoid",
                                            recurrent_initializer="glorot_uniform",
                                            stateful=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x):
        embedding = self.embedding(x)
        output = self.gru(embedding)
        prediction = self.fc(output)

        return prediction

def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


if __name__ == '__main__':

    path_to_file = tf.keras.utils.get_file("shakespeare.txt", 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    text = open(path_to_file).read()

    vocab = sorted(set(text))
    print(f"{len(vocab)} unique characters")

    char2idx = {u:i for i,u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    print(f"{text[:13]} ---> {text_as_int[:13]}")

    seq_length = 100
    chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)
    dataset = chunks.map(split_input_target)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(vocab)
    embedding_dim = 256
    units = 1024

    model = Model(vocab_size, embedding_dim, units)
    optimizer = tf.train.AdamOptimizer()
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
    model.summary()

    import pdb; pdb.set_trace()

    EPOCHS = 30

    for epoch in range(EPOCHS):
        start = time.time()
        hidden = model.reset_states()
        for batch, (inp, target) in enumerate(dataset):
            with tf.GradientTape() as tape:
                predictions = model(inp)
                loss = loss_function(target, predictions)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 100 == 0:
                print(f"Epoch {epoch+1} Batch {batch} Loss {loss}")
        if (epoch+1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        result_time = time.time() - start
        print(f"Epoch {epoch+1} Loss {loss}")
        print(f"Time taken for 1 epoch {result_time}")

    import pdb; pdb.set_trace()

