# Extract camera images from a specific angle
if __name__ == '__main__':
    import shutil
    my_src = '/Users/zhuoyuelyu/Downloads/Output-Oct-28-with-room-number-and-no-bookshelf-problem-png/'
    my_des = '/Users/zhuoyuelyu/Downloads/Output-Oct-28-with-room-number-and-no-bookshelf-problem-png-0/'
    for i in range(124):
        # for j in range(20):
        # print(i, j)
        shutil.copy2(my_src + str(i) + '-0-rgb.png', my_des)  # target filename is /dst/dir/file.ext
