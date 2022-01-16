import config
import torch
import sys

device = torch.device('cpu')


''' IMPORTANT:
  The model maps sensors to chunk features as
  [sensor_0, sensor_1, ...] => [chunk_0_feature_dim_0, chunk_0_feature_dim_1, ... chunk_1_feature_dim_0, chunk_1_feature_dim_1, ...]
'''
class Model(torch.nn.Module):
  def __init__(self, sensors, chunks, feature_dims):
    super().__init__()
    self.linear = torch.nn.Sequential(
      torch.nn.Linear(sensors, chunks * feature_dims),
      ## the activation layer is not necessarily required for one layer mapping
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


def compute_loss(pred, gt):
  diff = abs(pred - gt)
  ''' 
    Illuminance changes are much more common in a chunk than geometry,
    and thus, for reduction, we use mean for illuminance and max for geometry,
    encouraging the network to consistently pay attention to illuminance prediction,
    while also keep an eye on special geometry changes.
  '''
  geo_loss = torch.max(diff[0::config.feature_dims])
  lumin_loss = torch.mean(diff[config.feature_dims//2::config.feature_dims])
  loss = geo_loss + lumin_loss
  ## the old way treats geometry and illuminance equally
  # loss = torch.nn.L1Loss(reduction = 'mean')(pred, gt)
  return loss


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

  dataset_name = config.dataset_prefix + '.' + sys.argv[1]
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

  n_sensors = train_data[0][0].shape[0]
  n_chunks = train_data[0][1].shape[0] // config.feature_dims

  model = Model(n_sensors, n_chunks, config.feature_dims)
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-3)
  # optimizer = torch.optim.SGD(model.parameters(), lr = config.lr)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, config.min_lr)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

  for epoch in range(1, config.epochs + 1):
    train()
    if epoch % config.eval_freq == 0:
      eval()
      if epoch % config.save_freq == 0:
        torch.save(model.state_dict(), dataset_name + '/model.pth')