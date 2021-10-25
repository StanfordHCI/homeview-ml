import config
import torch
import sys

device = torch.device('cpu')


### Part 1: read data

def collate(datas):
  return [torch.stack(list(tup), 0) for tup in zip(*datas)]

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



### Part 2: define, init model

class Model(torch.nn.Module):
  def __init__(self, sensors, chunks):
    super().__init__()
    self.linear = torch.nn.Sequential(
      torch.nn.Linear(sensors, chunks),
      torch.nn.ReLU(),
    )
  def forward(self, input):
    output = self.linear(input)
    return output

n_sensors = train_data[0][0].shape[0]
n_chunks = train_data[0][1].shape[0]
# n_sensors = 1
# n_chunks = 1

model = Model(n_sensors, n_chunks)
model.to(device)


def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    torch.nn.init.uniform_(m.weight)
    torch.nn.init.uniform_(m.bias)

resume = 0
if resume:
  ckpt = torch.load(dataset_name + '/.pth')
  model.load_state_dict(ckpt)
else:
  # pass
  model.apply(init_weights)



### Part 3: loss and hyperparams

def compute_loss(pred, gt):
  loss = torch.nn.L1Loss(reduction = 'mean')(pred, gt)
  return loss


optimizer = torch.optim.SGD(model.parameters(), lr = config.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, config.min_lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)



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
  for epoch in range(1, config.epochs + 1):
    train()
    if epoch % config.eval_freq == 0:
      eval()
      if epoch % config.save_freq == 0:
        torch.save(model.state_dict(), dataset_name + '/.pth')