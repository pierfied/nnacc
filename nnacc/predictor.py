import torch
from torch import nn
from .nn import ResBlock
from tqdm.auto import tqdm


class Predictor:
    def __init__(self, in_size, out_size, model=None, optim=None, X_transform=None,
                 y_transform=None, device='cpu'):
        self.in_size = in_size
        self.out_size = out_size
        self.device = device

        if model is not None:
            self.model = model.to(device)
        else:
            hidden_size = max(in_size, out_size)
            self.model = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                ResBlock(hidden_size, hidden_size),
                ResBlock(hidden_size, hidden_size),
                ResBlock(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size)
            ).to(device)

        if optim is not None:
            self.optim = optim
        else:
            self.optim = torch.optim.Adam(self.model.parameters())

        if X_transform is not None:
            self.X_transform = X_transform
        else:
            self.X_transform = lambda x: x

        if y_transform is not None:
            self.y_transform = y_transform
        else:
            self.y_transform = lambda x: x

    def train(self, dataset, num_epochs, loss_fn, val_dataset=None, val_metric_fn=None):
        train_losses = []
        val_metrics = []

        if val_metric_fn is None:
            val_metric_fn = loss_fn

        pbar = tqdm(range(num_epochs))
        for i in pbar:
            data_iter = iter(dataset)

            self.model.train()

            for X, y_target in data_iter:
                X = X.to(self.device)
                y_target = y_target.to(self.device)

                self.optim.zero_grad()

                y_pred = self.y_transform(self.model(self.X_transform(X)))
                loss = loss_fn(y_pred, y_target)

                loss.backward()
                self.optim.step()

                train_losses.append(loss)
                pbar.set_description('Batch Loss: {}    Epoch'.format(loss))

            if val_dataset is not None:
                val_iter = iter(val_dataset)

                self.model.eval()

                val_metric = None
                val_count = 0
                for X, y_target in val_iter:
                    X = X.to(self.device)
                    y_target = y_target.to(self.device)

                    with torch.no_grad():
                        y_pred = self.y_transform(self.model(self.X_transform(X)))

                        if val_metric is None:
                            val_metric = val_metric_fn(y_pred, y_target)
                        else:
                            val_metric += val_metric_fn(y_pred, y_target)

                        val_count += 1

                val_metrics.append(val_metric / val_count)

        if val_dataset is not None:
            return torch.stack(train_losses).detach().to('cpu').numpy(),\
                   torch.stack(val_metrics).detach().to('cpu').numpy()
        else:
            return torch.stack(train_losses).detach().to('cpu').numpy()

    def predict(self, X):
        self.model.eval()

        if (len(X.shape) == 1):
            X = X.view(1, -1)
            one_input = True
        else:
            one_input = False

        with torch.no_grad():
            y_pred = self.y_transform(self.model(self.X_transform(X)))

        if one_input:
            y_pred = y_pred.view(-1)

        return y_pred
