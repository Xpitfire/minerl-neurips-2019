import torch


class CnnNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CnnNet, self).__init__()
        self.device = kwargs['device']

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0).to(self.device)
        self.conv4 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0).to(self.device)
        self.phi = torch.relu

    def forward(self, x):
        h = self.phi(self.conv1(x))
        h = self.phi(self.conv2(h))
        h = self.phi(self.conv3(h))
        h = self.phi(self.conv4(h))
        return h


class CnnLstmNet(torch.nn.Module):
    def __init__(self, cnn, lstm_hidden_dim, lstm_layer_dim, **kwargs):
        super(CnnLstmNet, self).__init__()
        self.device = kwargs['device']

        self.in_dim = 576
        self.cnn = cnn
        self.hidden_dim = lstm_hidden_dim
        self.layer_dim = lstm_layer_dim
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
    def __init__(self, lstm_hidden_dim, action_out_dim, **kwargs):
        super(ActionNet, self).__init__()
        self.device = kwargs['device']

        self.out_dim = action_out_dim
        self.hidden_dim = lstm_hidden_dim
        self.final = torch.nn.Linear(self.hidden_dim, self.out_dim).to(self.device)

    def forward(self, x):
        return self.final(x)


class CameraNet(torch.nn.Module):
    def __init__(self, lstm_hidden_dim, camera_out_dim, **kwargs):
        super(CameraNet, self).__init__()
        self.device = kwargs['device']

        self.out_dim = camera_out_dim
        self.hidden_dim = lstm_hidden_dim
        self.final = torch.nn.Linear(self.hidden_dim, self.out_dim).to(self.device)

    def forward(self, x):
        return self.final(x)


class Net(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.cnn = CnnNet(**kwargs.copy())
        self.cnn_lstm = CnnLstmNet(self.cnn, **kwargs.copy())
        self.action_net = ActionNet(**kwargs.copy())
        self.camera_net = CameraNet(**kwargs.copy())

    def forward(self, x):
        h, _ = self.cnn_lstm(x)
        h = h[:, -1, ...]
        action_out = self.action_net(h)
        camera_out = self.camera_net(h)
        return action_out, camera_out
