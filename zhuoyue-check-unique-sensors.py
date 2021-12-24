# check unique sensors
def get_comb(path):
    my_set = []
    dataset_name = "vh.cameras"
    train_data = torch.load(dataset_name + path)
    for i in range(len(train_data)):
        my_ele = train_data[i][0].tolist()
        for xxx in range(len(my_ele)):
            my_ele[xxx] = int(my_ele[xxx])
        if my_ele not in my_set:
            my_set.append(my_ele)
        # print(train_data[i][0].tolist())
    print(path)
    print(my_set)
    return my_set

if __name__ == '__main__':
    import itertools

    n = 4
    # the following will give us all unique binary combinartion
    comb = [list(i) for i in itertools.product([0, 1], repeat=n)]
    print(comb)
    my_set_train = []
    my_set_eval=[]

    my_set_train.extend(get_comb('/train-4-sensors.pth'))
    my_set_eval.extend(get_comb('/eval-4-sensors.pth'))

    # remove duplication in each other
    for x in my_set_eval:
        my_set_train.remove(x)
    # combine
    my_set_train.extend(my_set_eval)
    for x in my_set_train:
        comb.remove(x)
    print("remained")
    print(comb)