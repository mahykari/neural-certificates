import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D


N = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.ReLU())


def mse_loss(x, y):
  return F.mse_loss(x, y)


X = torch.Tensor([
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2]]) + 0.5

Y = torch.Tensor([
    [1, 0],
    [0, 2],
    [1, 2],
    [0, 0],
    [1, 0],
    [2, 2],
    [1, 0],
    [2, 2],
    [2, 1]]) + 0.5

S = torch.cat((X, Y), dim=1)

print('X =', X)
print('Y =', Y)
print('S =', S)


def loss(S):
  X, Y = S[:, :2], S[:, 2:]
  loss_eq = torch.unsqueeze(
      torch.norm(N(X) - N(Y), p=2, dim=1), dim=1)
  loss_gt = F.relu(N(Y) - N(X) + 1)
  sel = torch.zeros((9, 1))
  sel[torch.Tensor([0, 3, 7, 8]).int()] = 1
  # print(loss_eq * sel)
  # print(loss_gt * (1 - sel))
  return torch.sum(loss_eq * sel + loss_gt * (1 - sel))


def chk(S):
  X, Y = S[:, :2], S[:, 2:]
  chk_eq = N(X) == N(Y)
  chk_gt = N(X) - N(Y) >= 1
  sel = torch.zeros((9, 1))
  sel[torch.Tensor([0, 3, 7, 8]).int()] = 1
  return torch.mean(chk_eq * sel + chk_gt * (1 - sel)) * 100


def fit(S, n_epoch=512, batch_size=100, lr=1e-3,
        step_size=1, gamma=1.0):
  """Fits V based on a predefined loss function.

  Args:
    S: a set of sampled points from the state space.
  """

  optimizer = optim.SGD(N.parameters(), lr=lr)
  # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  scheduler = optim.lr_scheduler.StepLR(
      optimizer, step_size=step_size, gamma=gamma)
  el = []

  def training_step(s, optimizer):
    optimizer.zero_grad()
    loss_ = loss(s)
    chk_ = chk(s)
    loss_.backward(retain_graph=True)
    optimizer.step()
    return loss_, chk_

  def training_loop() -> None:
    n_batch = (len(S) + batch_size - 1) // batch_size
    assert batch_size * n_batch >= len(S)
    loader = D.DataLoader(S, batch_size=batch_size)
    step = 0
    for e in range(n_epoch + 1):
      epoch_loss = 0
      for b_idx, s in enumerate(loader):
        loss, _ = training_step(s, optimizer)
        epoch_loss += loss
        step += 1
      el.append(epoch_loss.detach())
      chk_ = chk(S)
      print(
          f'Epoch {e:>4} (Step {step:>6}), '
          + f'Epoch loss={epoch_loss:12.6f}, '
          + f'Chk={chk_:12.6f}, '
          + f'LR={optimizer.param_groups[0]["lr"]:13.8f}, ',
      )
      scheduler.step()
      if chk_ == 100:
        return

  training_loop()
  return el


fit(S, n_epoch=50000, batch_size=9, lr=.0002,
    gamma=1 - 1 / 500, step_size=100)
