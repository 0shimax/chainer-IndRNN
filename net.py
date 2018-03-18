import chainer
import chainer.functions as F
import chainer.links as L
import numpy


class IndRNN(chainer.Link):
    def __init__(self, num_input_units, num_units,
                 recurrent_max_abs=25., non_linearity=F.relu):
        super().__init__()
        recurrent_max_abs = min(recurrent_max_abs, 1.0 / numpy.sqrt(num_units))
        recurrent_initializer = chainer.initializers.Uniform(recurrent_max_abs)
        with self.init_scope():
            self.input_kernel = chainer.Parameter(
                None, (num_input_units, num_units))
            self.recurrent_kernel = chainer.Parameter(
                recurrent_initializer, (num_units,))
            self.bias = chainer.Parameter(0, (num_units,))
        self.non_linearity = non_linearity
        self.num_units = num_units
        self.recurrent_max_abs = recurrent_max_abs
        self.reset_state()

    def reset_state(self):
        self.h = chainer.Variable(
            self.xp.zeros(self.num_units, dtype=self.xp.float32))

    def set_hidden_state(self, state):
        self.h = state

    def clip_recurrent_kernel(self):
        self.recurrent_kernel.data = self.xp.clip(self.recurrent_kernel.data,
                                                  -self.recurrent_max_abs,
                                                  self.recurrent_max_abs)

    def __call__(self, x):
        gate_inputs = F.matmul(x, self.input_kernel)

        self.clip_recurrent_kernel()
        recurrent_update = self.h * \
            F.broadcast_to(self.recurrent_kernel, self.h.shape)
        gate_inputs = gate_inputs + \
            F.broadcast_to(recurrent_update, gate_inputs.shape)
        gate_inputs = gate_inputs + \
            F.broadcast_to(self.bias, gate_inputs.shape)
        output = self.non_linearity(gate_inputs)
        return output


class EncDec(chainer.Chain):

    def __init__(self, in_size, h_unit_size, slot_size, memory_size):
        super().__init__()
        with self.init_scope():
            self.encorder = IndRNN(in_size, h_unit_size)
            self.decorder = IndRNN(in_size, h_unit_size)
            self.hidden_to_output = L.Linear(h_unit_size, in_size)
        self.is_train = True

    def reset_state(self):
        self.encorder.reset_state()
        self.decorder.reset_state()

    def train(self, xs, ys):
        for time_idx in reversed(range(1, xs.shape[1])):
            dec_h = self.decorder(xs[:, time_idx])
            out = self.hidden_to_output(dec_h)
            ys.append(out.data)
            self.loss += (xs[:, time_idx-1] - out)**2
        return ys

    def predict(self, xs, ys, out):
        for time_idx in reversed(range(1, xs.shape[1])):
            dec_h = self.decorder(out)
            out = self.hidden_to_output(dec_h)
            ys.append(out.data)
        return ys

    def __call__(self, xs):
        n_batch, n_times, dim_obs = xs.shape
        for time_idx in range(n_times):
            x = xs[:, time_idx]
            h = self.encorder(x)

        self.decorder.set_hidden_state(h)
        ys = []
        out = self.hidden_to_output(self.decorder.h)

        ys.append(out)
        self.loss = (xs[:, -1] - out)**2

        if chainer.config.train:
            self.out = self.train(xs, ys)
        else:
            self.out = self.predict(xs, ys, out)

        self.loss /= n_times
        self.loss = F.sum(self.loss)/n_batch
        chainer.report({'loss': self.loss}, self)
        return self.loss
