import torch
from torch import nn
from .nn import ResBlock
from tqdm.auto import tqdm


class Predictor:
    def __init__(self, in_size, out_size, model=None, optim=None, device='cpu'):
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

    def train(self, dataset, num_epochs, loss_fn, val_dataset=None):
        train_losses = []
        val_losses = []

        pbar = tqdm(range(num_epochs))
        for i in pbar:
            data_iter = iter(dataset)

            self.model.train()

            for X, y_target in data_iter:
                self.optim.zero_grad()

                y_pred = self.model(X)
                loss = loss_fn(y_pred, y_target)

                loss.backward()
                self.optim.step()

                train_losses.append(loss)
                pbar.set_description('Batch Loss: {}\t\tEpoch:'.format(loss))

            if val_dataset is not None:
                val_iter = iter(val_dataset)

                self.model.eval()

                val_loss = 0
                val_count = 0
                for X, y_target in val_iter:
                    with torch.no_grad():
                        y_pred = self.model(X)
                        val_loss += loss_fn(y_pred, y_target)
                        val_count += 1

                val_losses.append(val_loss / val_count)

        if val_dataset is not None:
            return train_losses, val_losses
        else:
            return train_losses

    def predict(self, X, transform=None):
        self.model.eval()

        if(len(X.shape) == 1):
            X = X.view(1,-1)
            one_input = True
        else:
            one_input = False

        with torch.no_grad():
            y_pred = self.model(X)

            if transform is not None:
                y_pred = transform(y_pred)

        if one_input:
            y_pred = y_pred.view(-1)

        return y_pred
