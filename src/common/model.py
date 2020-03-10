import torch
from lighter.decorator import config, device


class CnnNet(torch.nn.Module):
    @device
    def __init__(self, config):
        super(CnnNet, self).__init__()
        self.config = config
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.phi = torch.relu

    def forward(self, x):
        h = self.phi(self.conv1(x))
        h = self.phi(self.conv2(h))
        h = self.phi(self.conv3(h))
        h = self.phi(self.conv4(h))
        return h


class CnnLstmNet(torch.nn.Module):
    @device
    def __init__(self, cnn, config):
        super(CnnLstmNet, self).__init__()
        self.config = config
        self.in_dim = 576
        self.cnn = cnn
        self.hidden_dim = self.config.lstm_hidden_dim
        self.layer_dim = self.config.lstm_layer_dim
        self.lstm = torch.nn.LSTM(self.in_dim, self.hidden_dim, self.layer_dim,
                                  batch_first=True, dropout=0.5).to(self.device)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim).to(self.device).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.shape[0], self.hidden_dim).to(self.device).requires_grad_()
        features = []
        for t in range(x.shape[1]):
            h = self.cnn(x[:, t, ...])
            h = h.view(x.shape[0], -1)
            features.append(h)
        h = torch.stack(features, dim=1).to(self.device)
        return self.lstm(h, (h0.detach(), c0.detach()))


class ActionNet(torch.nn.Module):
    @device
    def __init__(self, config):
        super(ActionNet, self).__init__()
        self.config = config
        self.out_dim = self.config.action_out_dim
        self.hidden_dim = self.config.lstm_hidden_dim
        self.final = torch.nn.Linear(self.hidden_dim, self.out_dim).to(self.device)

    def forward(self, x):
        return self.final(x)


class CameraNet(torch.nn.Module):
    @device
    def __init__(self, config):
        super(CameraNet, self).__init__()
        self.config = config
        self.out_dim = self.config.camera_out_dim
        self.hidden_dim = self.config.lstm_hidden_dim
        self.final = torch.nn.Linear(self.hidden_dim, self.out_dim).to(self.device)

    def forward(self, x):
        return self.final(x)


class ValueNet(torch.nn.Module):
    @device
    def __init__(self, config):
        super(ValueNet, self).__init__()
        self.config = config
        self.out_dim = self.config.value_out_dim
        self.hidden_dim = self.config.lstm_hidden_dim
        self.final = torch.nn.Linear(self.hidden_dim, self.out_dim).to(self.device)

    def forward(self, x):
        return self.final(x)


class Net(torch.nn.Module):
    @device
    @config(path="configs/config.model.json")
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = CnnNet()
        self.cnn_lstm = CnnLstmNet(self.cnn,
                                   self.config)
        self.action_net = ActionNet(self.config)
        self.camera_net = CameraNet(self.config)
        self.value_net = ValueNet(self.config)

    def forward(self, x):
        h, _ = self.cnn_lstm(x)
        h = h[:, -1, ...]
        action_out = self.action_net(h)
        camera_out = self.camera_net(h)
        value_out = self.value_net(h)
        return action_out, camera_out, value_out
