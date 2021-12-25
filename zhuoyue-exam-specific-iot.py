# Exam the states of a specific IoT
if __name__ == '__main__':
    import json

    dataset_name = 'output-dec-25'
    my_src = '/Users/zhuoyuelyu/Downloads/' + dataset_name + '/'
    # for i in range(171):
    #     f = open("{}/{}.json".format(my_src, i),)
    #     data = json.load(f)
    #     for entry in data:
    #         if entry['id'] == 402:
    #             print(entry['state'])
    count = 0

    """
    
    """
    for i in range(165):
        f = open("{}/{}.json".format(my_src, i), )
        data = json.load(f)
        for entry in data:
            # if entry['class_name'] == 'door' or entry['class_name'] == 'light':
            if entry['class_name'] == 'door' and entry['id'] == 305:
                print(i)
                print(entry['id'])
                print(entry['state'])
                print()
            # if entry['id'] == 305:
            #     print(entry['state'])
                # print()
