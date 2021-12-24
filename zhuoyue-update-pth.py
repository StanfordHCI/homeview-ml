import torch

# update pth
if __name__ == '__main__':
    indices = [0, 1, 2, 4, 6, 7]
    dataset_name = "vh.cameras"
    train_data = torch.load(dataset_name + '/train.pth')
    for i in range(len(train_data)):
        new_x = torch.tensor([train_data[i][0][index] for index in indices])
        train_data[i] = tuple([new_x, train_data[i][1]])
    torch.save(train_data, "train-6-sensors.pth")


