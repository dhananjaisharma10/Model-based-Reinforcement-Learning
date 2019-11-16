import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
from util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400


class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)

        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # TODO write your code here
        # Create and initialize your model
        self.inputs = []
        self.targets = []
        self.optimizations = []
        for model in range(self.num_nets):
            model, inp = self.create_network()
            self.inputs.append(inp)
            mean, logvar = self.get_output(model.output)
            target = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            self.targets.append(target)
            var = tf.math.exp(logvar)
            norm_output = mean - model.output[:, :self.state_dim]
            loss = tf.reshape(norm_output, shape=(1, -1)) * var * norm_output
            loss += tf.linalg.det(tf.diag(logvar))
            optimizer = Adam(lr=0.001)
            weights = model.trainable_weights
            gradients = tf.gradients(loss, weights)
            optimize = optimizer.apply_gradients(zip(gradients, weights))
            # optimize = optimizer.minimize(loss, weights)
            self.optimizations.append(optimize)

    def get_output(self, output):
        """
        Argument:
            output: tf variable representing the output of the keras models,
            i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in
        order to get the actual output.
        """
        mean = output[:, :self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        inp = Input(shape=[self.state_dim + self.action_dim])
        h1 = Dense(HIDDEN1_UNITS, activation='relu',
                   kernel_regularizer=l2(0.0001))(inp)
        h2 = Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu',
                   kernel_regularizer=l2(0.0001))(h2)
        out = Dense(2 * self.state_dim, activation='linear',
                    kernel_regularizer=l2(0.0001))(h3)
        model = Model(inputs=inp, outputs=out)
        return model, inp

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
            inputs: state and action inputs.  Assumes that inputs are
            standardized.
            targets: resulting states
        """
        # Sample 1000 episodes/transitions from inputs for each network.
        indices = [np.random.choice(range(len(inputs)),
                                    size=len(inputs),
                                    replace=True)
                   for x in range(self.num_nets)]
        for epoch in range(epochs):
            num_batches = math.ceil(len(inputs) / batch_size)
            for batch in range(num_batches):
                inps = [inputs[indices[x]] for x in range(self.num_nets)]
                targs = [targets[indices[x]] for x in range(self.num_nets)]
                losses = self.sess.run(self.optimizations,
                                       feed_dict={self.inputs: inps,
                                                  self.targets: targs})
                rmse_losses = [np.sqrt(loss / batch_size) for loss in losses]
