import config
import torch
import sys

device = torch.device('cpu')


### Part 1: define model

class Model(torch.nn.Module):
  def __init__(self, sensors, chunks, vector_dims):
    super().__init__()
    self.linear = torch.nn.Sequential(
      torch.nn.Linear(sensors, chunks * vector_dims),
      torch.nn.ReLU(),
    )
  def forward(self, input):
    output = self.linear(input)
    return output


def collate(datas):
  return [torch.stack(list(tup), 0) for tup in zip(*datas)]


def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    torch.nn.init.uniform_(m.weight)
    torch.nn.init.uniform_(m.bias)


### Part 3: loss and hyperparams

def compute_loss(pred, gt):
  loss = torch.nn.L1Loss(reduction = 'mean')(pred, gt)
  return loss


### Part 4: training and evaluation

def train():

  model.train()
  losses = []

  for batch_id, (input, output) in enumerate(train_loader):
    predict = model(input)
    loss = compute_loss(predict, output)
    losses.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

  print('[TRAIN] epoch {}, batch {}, loss: {:4f}'.format(epoch, batch_id + 1, sum(losses) / len(losses)))


def eval():

  model.eval()
  losses = []

  for batch_id, (input, output) in enumerate(eval_loader):
    predict = model(input)
    loss = compute_loss(predict, output)
    losses.append(loss)

  print('[EVAL] epoch {}, batch {}, loss: {:4f}'.format(epoch, batch_id + 1, sum(losses) / len(losses)))



if __name__ == '__main__':

  ### Part 2: read data

  dataset_name = 'vh.' + sys.argv[1]
  n_threads = 0

  train_data = torch.load(dataset_name + '/train.pth')
  train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size = config.batch_size,
                                            num_workers = n_threads,
                                            collate_fn = collate,
                                            shuffle = True,
                                            pin_memory = True)

  eval_data = torch.load(dataset_name + '/eval.pth')
  eval_loader = torch.utils.data.DataLoader(eval_data,
                                            batch_size = config.batch_size,
                                            num_workers = n_threads,
                                            collate_fn = collate,
                                            shuffle = False,
                                            pin_memory = True)

  print(train_data[0][1].shape, train_data[1][1].shape)
  n_sensors = train_data[0][0].shape[0]
  n_chunks = train_data[0][1].shape[0] / config.vector_dims
  # n_sensors = 1
  # n_chunks = 1


  model = Model(n_sensors, n_chunks, config.vector_dims)
  model.to(device)

  resume = 0
  if resume:
    ckpt = torch.load(dataset_name + '/.pth')
    model.load_state_dict(ckpt)
  else:
    # pass
    model.apply(init_weights)

  
  optimizer = torch.optim.SGD(model.parameters(), lr = config.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, config.min_lr)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


  for epoch in range(1, config.epochs + 1):
    train()
    if epoch % config.eval_freq == 0:
      eval()
      if epoch % config.save_freq == 0:
        torch.save(model.state_dict(), dataset_name + '/.pth')