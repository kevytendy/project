import torch
import torch.nn.functional as F
import torch.nn as nn
from opacus.validators import ModuleValidator  # For fixing BatchNorm layers

class VGAN_generator(nn.Module):
    """
    The generator for vanilla GAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, z_dim, hidden_dim, x_dim, layers, col_type, col_ind, condition=False, c_dim=0):
        super(VGAN_generator, self).__init__()
        self.input = nn.Linear(z_dim + c_dim, hidden_dim)
        self.inputbn = nn.GroupNorm(1, hidden_dim)  # Replaced BatchNorm with GroupNorm
        self.hidden = []
        self.BN = []
        self.col_type = col_type
        self.col_ind = col_ind
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.condition = condition
        for i in range(layers):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%d" % i, fc)
            self.hidden.append(fc)
            bn = nn.GroupNorm(1, hidden_dim)  # Replaced BatchNorm with GroupNorm
            setattr(self, "bn%d" % i, bn)
            self.BN.append(bn)
        self.output = nn.Linear(hidden_dim, x_dim)
        self.outputbn = nn.GroupNorm(1, x_dim)  # Replaced BatchNorm with GroupNorm

    def forward(self, z, c=None):
        if self.condition:
            assert c is not None
            z = torch.cat((z, c), dim=1)
        z = self.input(z)
        z = self.inputbn(z)
        z = torch.relu(z)
        for i in range(len(self.hidden)):
            z = self.hidden[i](z)
            z = self.BN[i](z)
            z = torch.relu(z)
        x = self.output(z)
        x = self.outputbn(x)
        # Normalize according to the column type
        output = []
        for i in range(len(self.col_type)):
            sta = self.col_ind[i][0]
            end = self.col_ind[i][1]
            if self.col_type[i] == 'binary':
                temp = torch.sigmoid(x[:, sta:end])
            elif self.col_type[i] == 'normalize':
                temp = torch.tanh(x[:, sta:end])
            elif self.col_type[i] == 'one-hot':
                temp = torch.softmax(x[:, sta:end], dim=1)
            elif self.col_type[i] == 'gmm':
                temp1 = torch.tanh(x[:, sta:sta + 1])
                temp2 = torch.softmax(x[:, sta + 1:end], dim=1)
                temp = torch.cat((temp1, temp2), dim=1)
            elif self.col_type[i] == 'ordinal':
                temp = torch.sigmoid(x[:, sta:end])
            output.append(temp)
        output = torch.cat(output, dim=1)
        return output


class VGAN_discriminator(nn.Module):
    """
    The discriminator for vanilla GAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, x_dim, hidden_dim, layers, condition=False, c_dim=0, wgan=False):
        super(VGAN_discriminator, self).__init__()
        self.input = nn.Linear(x_dim + c_dim, hidden_dim)
        self.hidden = []
        self.Dropout = nn.Dropout(p=0.5)
        self.condition = condition
        self.wgan = wgan
        for i in range(layers):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%d" % i, fc)
            self.hidden.append(fc)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, z, c=None):
        if self.condition:
            assert c is not None
            z = torch.cat((z, c), dim=1)
        z = self.input(z)
        z = torch.relu(z)
        z = self.Dropout(z)
        for i in range(len(self.hidden)):
            z = self.hidden[i](z)
            z = torch.relu(z)
            z = self.Dropout(z)
        z = self.output(z)
        if self.wgan:
            return z
        else:
            return torch.sigmoid(z)


class LSTM_discriminator(nn.Module):
    """
    The discriminator for LSTM-GAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, x_dim, lstm_dim, condition=False, c_dim=0):
        super(LSTM_discriminator, self).__init__()
        self.LSTM = nn.LSTMCell(1 + c_dim, lstm_dim)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_dim, 1),
            nn.Sigmoid()
        )
        self.condition = condition
        self.l_dim = lstm_dim

    def forward(self, x, c=None):
        if self.condition:
            assert c is not None
        hx = torch.randn(x.size(0), self.l_dim)
        cx = torch.randn(x.size(0), self.l_dim)
        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()
        for i in range(x.size(1)):
            input_x = x[:, i].reshape(-1, 1)
            if self.condition:
                input_x = torch.cat((input_x, x), dim=1)
            hx, cx = self.LSTM(input_x, (hx, cx))
        return self.mlp(hx)


class LGAN_generator(nn.Module):
    """
    The generator for LSTM-GAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, z_dim, feature_dim, lstm_dim, col_dim, col_type, condition=False, c_dim=0):
        super(LGAN_generator, self).__init__()
        self.condition = condition
        self.c_dim = c_dim
        self.l_dim = lstm_dim
        self.f_dim = feature_dim
        self.col_dim = col_dim
        self.col_type = col_type
        self.GPU = False
        self.LSTM = nn.LSTMCell(z_dim + c_dim + feature_dim, lstm_dim)  # input (fx, z, attention), output(hx, cx)
        self.FC = {}  # FullyConnect layers for every column
        self.Feature = {}
        self.go = nn.Parameter(torch.randn(1, self.f_dim))
        for i in range(len(col_type)):
            if col_type[i] == "condition":
                continue
            if col_type[i] == "gmm":
                self.FC[i] = []
                fc1 = nn.Linear(feature_dim, 1)
                setattr(self, "gmfc%d0" % i, fc1)

                fc2 = nn.Linear(feature_dim, col_dim[i] - 1)
                setattr(self, "gmfc%d1" % i, fc2)

                fc3 = nn.Linear(col_dim[i] - 1, feature_dim)
                setattr(self, "gmfc%d2" % i, fc3)
                self.FC[i] = [fc1, fc2, fc3]

                fe1 = nn.Linear(lstm_dim, feature_dim)
                setattr(self, "gmfe%d0" % i, fe1)

                fe2 = nn.Linear(lstm_dim, feature_dim)
                setattr(self, "gmfe%d1" % i, fe2)
                self.Feature[i] = [fe1, fe2]
            else:
                fc1 = nn.Linear(feature_dim, col_dim[i])
                setattr(self, "fc%d0" % i, fc1)

                fc2 = nn.Linear(col_dim[i], feature_dim)
                setattr(self, "fc%d1" % i, fc2)
                self.FC[i] = [fc1, fc2]

                fe = nn.Linear(lstm_dim, feature_dim)
                setattr(self, "fe%d" % i, fe)
                self.Feature[i] = fe

    def forward(self, z, c=None):
        if self.condition:
            assert c is not None
            z = torch.cat((z, c), dim=1)
        hx = torch.randn(z.size(0), self.l_dim)
        cx = torch.randn(z.size(0), self.l_dim)
        fx = self.go.repeat([z.size(0), 1])
        if self.GPU:
            hx = hx.cuda()
            cx = cx.cuda()
            fx = fx.cuda()
        inputs = torch.cat((z, fx), dim=1)
        outputs = []
        for i in range(len(self.col_type)):
            if self.col_type[i] == "condition":
                continue
            if self.col_type[i] == "gmm":
                hx, cx = self.LSTM(inputs, (hx, cx))
                fx = torch.tanh(self.Feature[i][0](hx))
                v = torch.tanh(self.FC[i][0](fx))
                outputs.append(v)
                inputs = torch.cat((z, fx), dim=1)

                hx, cx = self.LSTM(inputs, (hx, cx))
                fx = torch.tanh(self.Feature[i][1](hx))
                v = self.FC[i][1](fx)
                v = torch.softmax(v, dim=1)
                outputs.append(v)
                fx = torch.tanh(self.FC[i][2](v))
                inputs = torch.cat((z, fx), dim=1)
            else:
                hx, cx = self.LSTM(inputs, (hx, cx))
                fx = self.Feature[i](hx)
                v = self.FC[i][0](fx)
                if self.col_type[i] == "binary":
                    v = torch.sigmoid(v)
                elif self.col_type[i] == "normalize":
                    v = torch.tanh(v)
                elif self.col_type[i] == "one-hot":
                    v = torch.softmax(v, dim=1)
                elif self.col_type[i] == "ordinal":
                    v = torch.sigmoid(v)
                else:
                    v = F.leaky_relu(v)
                outputs.append(v)
                fx = self.FC[i][1](v)
                inputs = torch.cat((z, fx), dim=1)
        true_output = torch.cat(outputs, dim=1)
        return true_output


class LGAN_discriminator(nn.Module):
    """
    The discriminator of LSTM-GAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, x_dim, hidden_dim, num_layers, condition=False, c_dim=0, wgan=False):
        super(LGAN_discriminator, self).__init__()
        self.condition = condition
        self.BatchNorm = []
        self.Dropout = nn.Dropout(p=0.5)
        self.FC = []
        self.input = nn.Linear(x_dim + c_dim, hidden_dim)
        self.inputbn = nn.GroupNorm(1, hidden_dim)  # Replaced BatchNorm with GroupNorm
        self.wgan = wgan
        for i in range(num_layers):
            fc = nn.Linear(hidden_dim, hidden_dim)
            setattr(self, "fc%i" % i, fc)
            bn = nn.GroupNorm(1, hidden_dim)  # Replaced BatchNorm with GroupNorm
            setattr(self, "bn%i" % i, bn)
            self.BatchNorm.append(bn)
            self.FC.append(fc)

        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, c=None):
        if self.condition:
            assert c is not None
            x = torch.cat((x, c), dim=1)
        x = self.input(x)
        x = self.inputbn(x)
        for i, layer in enumerate(self.FC):
            x = layer(x)
            x = self.BatchNorm[i](x)
            x = self.Dropout(x)
            x = F.leaky_relu(x)
        x = self.output(x)
        if self.wgan:
            return x
        else:
            return torch.sigmoid(x)


class DCGAN_generator(nn.Module):
    """
    The generator of DCGAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, z_dim, shape, kernel, col_type):
        super(DCGAN_generator, self).__init__()
        in_channal = 4
        out_channal = 1
        kernel_size = kernel
        self.BN = []
        self.DV = []
        self.layer_num = 0
        self.shape = shape
        self.col_type = col_type
        self.GPU = False

        # Build the generator network
        while shape > kernel_size:
            deconv = nn.ConvTranspose2d(in_channal, out_channal, kernel_size, bias=False)
            setattr(self, "conv%d" % self.layer_num, deconv)
            self.DV.append(deconv)
            if self.layer_num == 0:
                self.BN.append([])
            else:
                bn = nn.GroupNorm(1, out_channal)  # Replaced BatchNorm with GroupNorm
                setattr(self, "bn%d" % self.layer_num, bn)
                self.BN.append(bn)
            out_channal = in_channal
            in_channal = in_channal * 2
            shape = (shape - kernel_size) + 1
            self.layer_num += 1

        # Input layer
        self.input = nn.ConvTranspose2d(z_dim, out_channal, shape, bias=False)
        self.bn = nn.GroupNorm(1, out_channal)  # Replaced BatchNorm with GroupNorm

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.input(x)
        x = self.bn(x)
        x = F.relu(x)

        # Apply transposed convolutions
        for i in range(len(self.DV) - 1, -1, -1):
            x = self.DV[i](x)
            if i != 0:
                x = self.BN[i](x)
                x = F.relu(x)

        # Normalize output according to column type
        for col, t in enumerate(self.col_type):
            i = int(col / self.shape)
            j = col % self.shape
            if t == "binary":
                x[:, :, i, j] = torch.sigmoid(x[:, :, i, j])
            elif t == "normalize":
                x[:, :, i, j] = torch.tanh(x[:, :, i, j])
            else:
                x[:, :, i, j] = torch.relu(x[:, :, i, j])

        return x


class DCGAN_discriminator(nn.Module):
    """
    The discriminator of DCGAN.
    Modified to support dynamic clipping and advanced DP mechanisms.
    """

    def __init__(self, shape, kernel):
        super(DCGAN_discriminator, self).__init__()
        in_channal = 1
        out_channal = 4
        kernel_size = kernel
        self.CV = []
        self.BN = []
        self.layer_num = 0
        self.shape = shape

        # Build the discriminator network
        while shape > kernel_size:
            conv = nn.Conv2d(in_channal, out_channal, kernel_size, bias=False)
            setattr(self, "conv%d" % self.layer_num, conv)
            self.CV.append(conv)
            bn = nn.GroupNorm(1, out_channal)  # Replaced BatchNorm with GroupNorm
            setattr(self, "bn%d" % self.layer_num, bn)
            self.BN.append(bn)
            in_channal = out_channal
            out_channal = out_channal * 2
            shape = (shape - kernel_size) + 1
            self.layer_num += 1

        # Output layer
        self.output = nn.Conv2d(in_channal, 1, shape, bias=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, self.shape, -1)

        # Apply convolutions
        for i in range(self.layer_num):
            x = self.CV[i](x)
            x = self.BN[i](x)
            x = F.leaky_relu(x)

        # Output layer
        x = self.output(x)
        x = torch.sigmoid(x)
        x = x.reshape(-1, 1)

        return x