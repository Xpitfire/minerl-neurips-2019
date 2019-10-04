import numpy as np


class Stats:
    def __init__(self, writer, **kwargs):
        self.writer = writer
        self._reset()

    def _reset_train_lists(self):
        self.train_loss = []
        self.train_action_loss = []
        self.train_camera_loss = []
        self.train_value_loss = []
        self.train_acc = []

    def _reset_val_lists(self):
        self.val_loss = []
        self.val_action_loss = []
        self.val_camera_loss = []
        self.val_value_loss = []
        self.val_acc = []

    def _reset(self):
        self._reset_train_lists()
        self._reset_val_lists()

        self._set_print_end('\n')
        self.train_loss_steps = 0
        self.train_action_loss_steps = 0
        self.train_camera_loss_steps = 0
        self.train_value_loss_steps = 0

        self.val_loss_steps = 0
        self.val_action_loss_steps = 0
        self.val_camera_loss_steps = 0
        self.val_value_loss_steps = 0

        self.train_acc_steps = 0
        self.val_acc_steps = 0


    def _set_print_end(self, end):
        self.end_symbol = end

    def add_train_loss(self, loss):
        self.train_loss.append(loss)
        self.writer.add_scalar('Loss/train', loss, self.train_loss_steps)
        self.train_loss_steps += 1

    def add_train_action_loss(self, loss):
        self.train_action_loss.append(loss)
        self.writer.add_scalar('Loss/action_train', loss, self.train_action_loss_steps)
        self.train_action_loss_steps += 1

    def add_train_camera_loss(self, loss):
        self.train_camera_loss.append(loss)
        self.writer.add_scalar('Loss/camera_train', loss, self.train_camera_loss_steps)
        self.train_camera_loss_steps += 1

    def add_train_value_loss(self, loss):
        self.train_value_loss.append(loss)
        self.writer.add_scalar('Loss/train_value', loss, self.train_value_loss_steps)
        self.train_value_loss_steps += 1

    def add_val_loss(self, loss):
        self.val_loss.append(loss)
        self.writer.add_scalar('Loss/val', loss, self.val_loss_steps)
        self.val_loss_steps += 1

    def add_val_action_loss(self, loss):
        self.val_action_loss.append(loss)
        self.writer.add_scalar('Loss/val_action', loss, self.val_action_loss_steps)
        self.val_action_loss_steps += 1

    def add_val_camera_loss(self, loss):
        self.val_camera_loss.append(loss)
        self.writer.add_scalar('Loss/val_camera', loss, self.val_camera_loss_steps)
        self.val_camera_loss_steps += 1

    def add_val_value_loss(self, loss):
        self.val_value_loss.append(loss)
        self.writer.add_scalar('Loss/val_value', loss, self.val_value_loss_steps)
        self.val_value_loss_steps += 1

    def add_train_acc(self, acc):
        self.train_acc.append(acc)
        self.writer.add_scalar('Accuracy/train', acc, self.train_acc_steps)
        self.train_acc_steps += 1

    def add_val_acc(self, acc):
        self.val_acc.append(acc)
        self.writer.add_scalar('Accuracy/val', acc, self.val_acc_steps)
        self.val_acc_steps += 1

    def print_mean_train_loss(self):
        print('mean train loss:', np.mean(self.train_loss), end=self.end_symbol)

    def print_mean_val_loss(self):
        print('mean val loss:', np.mean(self.val_loss), end=self.end_symbol)

    def print_mean_train_acc(self):
        print('mean train acc:', np.mean(self.train_acc), end=self.end_symbol)

    def print_mean_val_acc(self):
        print('mean val acc:', np.mean(self.val_acc), end=self.end_symbol)

    def print_train_stats(self):
        print('train loss: ', np.mean(self.train_loss),
              'train action loss: ', np.mean(self.train_action_loss),
              'train camera loss: ', np.mean(self.train_camera_loss),
              'train value loss: ', np.mean(self.train_value_loss),
              'train acc:', np.mean(self.train_acc),
              end=self.end_symbol)
        self._reset_train_lists()

    def print_val_stats(self):
        print('val loss: ', np.mean(self.val_loss),
              'val action loss: ', np.mean(self.val_action_loss),
              'val camera loss: ', np.mean(self.val_camera_loss),
              'val value loss: ', np.mean(self.val_value_loss),
              'val acc:', np.mean(self.val_acc),
              end=self.end_symbol)
        self._reset_val_lists()
