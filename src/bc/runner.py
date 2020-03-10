import time
import datetime
import pathlib
import traceback
from lighter.decorator import config, device
import torch
import os
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.bc.utils import set_seeds, count_parameters, save_model, load_model
from src.common.data import load_data, data_wrapper
from src.common.model import Net
from src.common.stats import Stats
from src.bc.agent import build_agent


class Runner:
    @device
    @config(path="configs/config.bc_test.json")
    def __init__(self):
        try:
            st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S')
            self.config.save_dir = os.path.join(self.save_dir, st)
            pathlib.Path(os.path.join(self.save_dir, st)).mkdir(parents=True, exist_ok=True)
            self.config.rec_save_dir = os.path.join(self.rec_save_dir, st)
            pathlib.Path(os.path.join(self.rec_save_dir, st)).mkdir(parents=True, exist_ok=True)
            self.config.save()
        except:
            traceback.print_exc()
            exit(-1)

    def get_criterion(self):
        return torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss(), torch.nn.MSELoss()

    def get_optimizer(self, params, lr, optimizer, momentum=None, **kwargs):
        if optimizer == 'adam':
            return optim.Adam(params, lr=lr)
        elif optimizer == 'sgd':
            if momentum is not None:
                return optim.SGD(params, lr=lr, momentum=momentum)
            else:
                return optim.SGD(params, lr=lr)
        return None

    def get_metric(self, metric, **kwargs):
        if metric == 'accuracy':
            def accuracy(y_pred, y_true):
                acc = ((torch.sigmoid(y_pred) > 0.5).float() == y_true).float()
                return torch.sum(acc, dim=-1).mean() / y_true.shape[-1]
            return accuracy
        return None

    def update(self, model, x, y, optimizer, criterion, metric, stats, grad_clip,
               camera_lambda, value_lambda, device):
        model.train()
        states = x
        actions, cameras, values = y

        inputs = torch.from_numpy(states[:, :-1, ...]).to(device)
        action_target = torch.from_numpy(actions[:, -1, :]).to(device)
        camera_target = torch.from_numpy(cameras[:, -1, :]).to(device)
        value_target = torch.from_numpy(values[:, -1, :]).to(device)

        # get predictions
        action_pred, camera_pred, value_pred = model(inputs)
        action_loss = criterion[0](action_pred, action_target)
        # TODO: camera_loss hack which scales down the loss; replace by target scaling
        camera_loss = camera_lambda * torch.log(criterion[1](camera_pred, camera_target) + 1.)
        value_loss = value_lambda * torch.log(criterion[2](value_pred, value_target) + 1.)
        loss = action_loss + camera_loss + value_loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # clip gradients before update
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # collect metrics
        stats.add_train_loss(loss.item())
        stats.add_train_action_loss(action_loss.item())
        stats.add_train_camera_loss(camera_loss.item())
        stats.add_train_value_loss(value_loss.item())

        acc = metric(action_pred, action_target)
        stats.add_train_acc(acc.item())

    def evaluate(self, model, dataloader, criterion, metric, stats, batch_size, seed,
                 sequence_len, camera_lambda, value_lambda, device):
        print('========= Evaluate policy ==========')
        model.eval()
        with torch.no_grad():
            # set a sampling maximum to speed up evaluation
            sample_size = 50
            samples = 0
            for states, actions, cameras, values in dataloader(1, batch_size, seed, sequence_len):
                inputs = torch.from_numpy(states[:, :-1, ...]).to(device)
                action_target = torch.from_numpy(actions[:, -1, :]).to(device)
                camera_target = torch.from_numpy(cameras[:, -1, :]).to(device)
                value_target = torch.from_numpy(values[:, -1, :]).to(device)

                # get predictions
                action_pred, camera_pred, value_pred = model(inputs)
                action_loss = criterion[0](action_pred, action_target)
                # TODO: camera_loss hack which scales down the loss; replace by target scaling
                camera_loss = camera_lambda * torch.log(criterion[1](camera_pred, camera_target) + 1.)
                value_loss = value_lambda * torch.log(criterion[2](value_pred, value_target) + 1.)
                loss = action_loss + camera_loss + value_loss

                # collect metrics
                stats.add_val_loss(loss.item())
                stats.add_val_action_loss(action_loss.item())
                stats.add_val_camera_loss(camera_loss.item())
                stats.add_val_value_loss(value_loss.item())

                acc = metric(action_pred, action_target)
                stats.add_val_acc(acc.item())

                if samples < sample_size:
                    break
                samples += 1

        stats.print_val_stats()

    def run(self, epochs, run_mode, batch_size, grad_clip, seed=None):
        only_env_seed = self.config.only_env_seed
        set_seeds(seed, only_env_seed)
        # assign local variable to avoid kwargs pop
        sequence_len = self.config.sequence_len
        camera_lambda = self.config.camera_lambda
        value_lambda = self.config.value_lambda
        writer = SummaryWriter(log_dir=os.path.join(self.config.save_dir, "tb"))

        train_dataloaders = []
        val_dataloader = None
        # prepare data
        if run_mode != 'agent':
            train_envs = self.config.train_env_names.split(',')
            for env_name in train_envs:
                train_dataloaders.append(data_wrapper(load_data(data_dir=self.config.train_data_dir,
                                                                env_name=env_name,
                                                                **kwargs.copy())))
            val_dataloader = data_wrapper(load_data(data_dir=self.config.val_data_dir,
                                                    env_name=self.config.val_env_name,
                                                    **kwargs.copy()))

        # model loading and optimizer definition
        model = Net()
        print('Trainable parameters:', count_parameters(model))
        load_model(model, **kwargs.copy())

        # optimization settings
        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model.parameters(), **kwargs.copy())

        # metric and stats
        metric = self.get_metric(**kwargs.copy())
        stats = Stats(writer, **kwargs.copy())

        # create test agent
        agent = build_agent(seed=seed)
        if run_mode == 'agent':
            agent.run(model)
            return

        # evaluate model
        self.evaluate(model=model, dataloader=val_dataloader, criterion=criterion,
                      metric=metric, stats=stats, batch_size=batch_size, seed=seed,
                      sequence_len=sequence_len, camera_lambda=camera_lambda,
                      value_lambda=value_lambda, device=device)
        if run_mode == 'eval':
            return

        # main train loop
        for e in range(epochs):
            print(' Epoch: {} / {}'.format(e+1, epochs))
            for dl in train_dataloaders:
                for states, actions, cameras, values in tqdm(dl(1, batch_size, seed, sequence_len)):
                    self.update(model=model, x=states, y=(actions, cameras, values),
                                optimizer=optimizer, criterion=criterion, metric=metric,
                                stats=stats, grad_clip=grad_clip, camera_lambda=camera_lambda,
                                value_lambda=value_lambda, device=device)

            # print metrics
            stats.print_train_stats()

            # periodically save the model state
            self.evaluate(model=model, dataloader=val_dataloader, criterion=criterion,
                          metric=metric, stats=stats, batch_size=batch_size, seed=seed,
                          sequence_len=sequence_len, camera_lambda=camera_lambda,
                          value_lambda=value_lambda, device=device)
            if agent is not None:
                agent.run(model)
            save_model(model, self.config.save_dir, e)
