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
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]),
                                      dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]),
                                      dtype=tf.float32)
        # Define ops for model output and optimization
        self.outs = list()
        self.inputs = list()
        self.losses = list()
        self.models = list()
        self.targets = list()
        self.optimizations = list()
        for model in range(self.num_nets):
            model, inp = self.create_network()
            self.inputs.append(inp)
            self.models.append(model)
            output = self.get_output(model.output)
            mean, logvar = output
            self.outs.append(output)
            target = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            self.targets.append(target)
            var = tf.math.exp(logvar)
            cov = tf.diag(var)
            norm_output = mean - model.output[:, :self.state_dim]
            # Calculate loss: Mahalanobis distance + log(det(cov))
            # NOTE: Shape of norm_output should be (-1, self.state_dim)
            loss = tf.math.multiply(tf.math.divide(norm_output, cov),
                                    tf.transpose(norm_output))
            loss += tf.math.log(tf.linalg.det(cov))
            self.losses.append(loss)
            optimizer = Adam(lr=0.001)
            weights = model.trainable_weights
            gradients = tf.gradients(loss, weights)
            optimize = optimizer.apply_gradients(zip(gradients, weights))
            self.optimizations.append(optimize)
        self.sess.run(tf.initialize_all_variables())

    def predict(self, state, action):
        input = np.concatenate((state, action), axis=1)
        feed_dict = {inp: input for inp in self.inputs}
        outputs = self.sess.run(self.outs, feed_dict=feed_dict)
        # TODO: use the output of all models
        output = outputs[0]
        mean, logvar = output
        sigma = np.sqrt(np.exp(logvar))
        state = np.random.normal(mean, sigma, size=mean.shape)
        return state

    def get_output(self, output):
        """
        Args:
            output: tf variable representing the output of the keras models,
            i.e., model.output

        Returns:
            mean and log variance tf tensors
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

    def get_indices(self, n):
        return np.random.choice(range(n), size=n, replace=True)

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Args:
            inputs: state and action inputs.  Assumes that inputs are
            standardized.
            targets: resulting states
        """
        # Sample indices with replacement for all models
        # NOTE: Refer to Piazza #805, #804 for RMSE calculation details.
        size = len(inputs)
        indices = [self.get_indices(size) for _ in range(self.num_nets)]
        total_loss = list()
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            num_batches = math.ceil(size / batch_size)
            for batch in range(num_batches):
                # Sample a batch and get inputs and targets
                idx = batch * batch_size
                real_batch_size = min(size - idx, batch_size)
                inps = [inputs[indices[x][idx:idx + real_batch_size]]
                        for x in range(self.num_nets)]
                targs = [targets[indices[x][idx:idx + real_batch_size]]
                         for x in range(self.num_nets)]
                feed_dict = {self.inputs: inps, self.targets: targs}
                # TODO: Add a list for getting the losses
                losses = self.sess.run(self.optimizations, feed_dict=feed_dict)
                summed_loss = sum(losses)
                print('Batch {}/{} | Loss: {:.3f}'.format(batch + 1,
                                                          num_batches,
                                                          summed_loss),
                      end='\r', flush=True)
                total_loss.append(sum(losses))
                # rmse_losses = [np.sqrt(loss / batch_size) for loss in losses]
        np.save('losses.npy', total_loss)
