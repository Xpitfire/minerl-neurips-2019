import torch
import os
import random
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.utils import set_seeds, count_parameters, save_model, load_model
from src.data import load_data, data_wrapper, transforms
from src.model import Net
from src.stats import Stats
from src.agent import build_agent


def get_model(**kwargs):
    return Net(**kwargs)


def get_criterion(**kwargs):
    return torch.nn.BCEWithLogitsLoss(), torch.nn.MSELoss()


def get_optimizer(params, lr, optimizer, momentum=None, **kwargs):
    if optimizer == 'adam':
        return optim.Adam(params, lr=lr)
    elif optimizer == 'sgd':
        if momentum is not None:
            return optim.SGD(params, lr=lr, momentum=momentum)
        else:
            return optim.SGD(params, lr=lr)
    return None


def get_metric(metric, **kwargs):
    if metric == 'accuracy':
        def accuracy(y_pred, y_true):
            acc = ((torch.sigmoid(y_pred) > 0.5).float() == y_true).float()
            return torch.sum(acc, dim=-1).mean() / y_true.shape[-1]
        return accuracy
    return None


def update(model, x, y, optimizer, criterion, metric, stats, grad_clip, device):
    model.train()
    states = x
    actions, cameras = y

    inputs = torch.from_numpy(states[:, :-1, ...]).to(device)
    action_target = torch.from_numpy(actions[:, -1, :]).to(device)
    camera_target = torch.from_numpy(cameras[:, -1, :]).to(device)

    # get predictions
    action_pred, camera_pred = model(inputs)
    action_loss = criterion[0](action_pred, action_target)
    camera_loss = criterion[1](camera_pred, camera_target)
    loss = action_loss + 0.1 * torch.log(camera_loss)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # clip gradients before update
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    # collect metrics
    stats.add_train_loss(loss.item())
    acc = metric(action_pred, action_target)
    stats.add_train_acc(acc.item())


def evaluate(model, dataloader, criterion, metric, stats,
             batch_size, seed, sequence_len, device):
    print('========= Evaluate policy ==========')
    model.eval()
    with torch.no_grad():
        # set a sampling maximum to speed up evaluation
        sample_size = 50
        samples = 0
        for states, actions, cameras in dataloader(1, batch_size, seed, sequence_len):
            inputs = torch.from_numpy(states[:, :-1, ...]).to(device)
            action_target = torch.from_numpy(actions[:, -1, :]).to(device)
            camera_target = torch.from_numpy(cameras[:, -1, :]).to(device)

            # get predictions
            action_pred, camera_pred = model(inputs)
            action_loss = criterion[0](action_pred, action_target)
            camera_loss = criterion[1](camera_pred, camera_target)
            loss = action_loss + 0.1 * torch.log(camera_loss)

            # collect metrics
            stats.add_val_loss(loss.item())
            acc = metric(action_pred, action_target)
            stats.add_val_acc(acc.item())

            if samples < sample_size:
                break
            samples += 1

    stats.print_val_stats()


def run(epochs, run_mode, batch_size, grad_clip, seed=None, **kwargs):
    only_env_seed = kwargs['only_env_seed']
    set_seeds(seed, only_env_seed)
    # assign local variable to avoid kwargs pop
    sequence_len = kwargs['sequence_len']
    device = kwargs['device']
    writer = SummaryWriter(log_dir=os.path.join(kwargs['save_dir'], "tb"))

    train_dataloaders = []
    val_dataloader = None
    # prepare data
    if run_mode != 'agent':
        train_envs = kwargs['train_env_names'].split(',')
        for env_name in train_envs:
            train_dataloaders.append(data_wrapper(load_data(data_dir=kwargs['train_data_dir'],
                                                            env_name=env_name,
                                                            **kwargs.copy()), transforms))
        val_dataloader = data_wrapper(load_data(data_dir=kwargs['val_data_dir'],
                                                env_name=kwargs['val_env_name'],
                                                **kwargs.copy()), transforms)

    # model loading and optimizer definition
    model = get_model(**kwargs.copy())
    print('Trainable parameters:', count_parameters(model))
    load_model(model, **kwargs.copy())

    # optimization settings
    criterion = get_criterion(**kwargs.copy())
    optimizer = get_optimizer(model.parameters(), **kwargs.copy())

    # metric and stats
    metric = get_metric(**kwargs.copy())
    stats = Stats(writer, **kwargs.copy())

    # create test agent
    agent = build_agent(seed=seed, **kwargs.copy())
    if run_mode == 'agent':
        agent.run(model, **kwargs.copy())
        return

    # evaluate model
    evaluate(model=model, dataloader=val_dataloader, criterion=criterion,
             metric=metric, stats=stats, batch_size=batch_size, seed=seed,
             sequence_len=sequence_len, device=device)
    if run_mode == 'eval':
        return

    # main train loop
    for e in range(epochs):
        print(' Epoch: {} / {}'.format(e+1, epochs))
        for dl in train_dataloaders:
            for states, actions, cameras in tqdm(dl(1, batch_size, seed, sequence_len)):

                update(model=model, x=states, y=(actions, cameras),
                       optimizer=optimizer, criterion=criterion, metric=metric,
                       stats=stats, grad_clip=grad_clip, device=device)

                # print metrics
                stats.print_train_stats()

        # periodically save the model state
        evaluate(model=model, dataloader=val_dataloader, criterion=criterion,
                 metric=metric, stats=stats, batch_size=batch_size, seed=seed,
                 sequence_len=sequence_len, device=device)
        if agent is not None:
            agent.run(model, **kwargs.copy())
        save_model(model, kwargs['save_dir'], e)
