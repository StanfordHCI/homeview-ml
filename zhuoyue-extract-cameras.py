# Extract camera images from a specific angle
def extrac_camera(my_camera, dataset_name, dataset_range):
    my_src = '/Users/zhuoyuelyu/Downloads/' + dataset_name + '/'
    my_des = '/Users/zhuoyuelyu/Downloads/' + dataset_name + '-' + str(my_camera) + '/'
    for i in range(dataset_range):
        if not os.path.isdir(my_des):
            os.mkdir(my_des)
        shutil.copy2(my_src + str(i) + '-' + str(my_camera) + '-rgb.png',
                     my_des)  # target filename is /dst/dir/file.ext


if __name__ == '__main__':
    import shutil
    import os

    """
    0: check the D1 and L1
    
    So we only need to the following to check everything
    5: check the D3, L2, L3
    14: check the D2, L5
    16: check the D1, D2, L1, L4
    """
    dataset_name = 'output-dec-25'
    dataset_range = 165
    for i in [5, 14, 16]:
        extrac_camera(i, dataset_name, dataset_range)
