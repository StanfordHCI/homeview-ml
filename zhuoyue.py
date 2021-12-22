# # convert all npz to ply
#
#
#
# def zhuoyue_convert_point_cloud(new_frame_id, chunk_id):
#     frame_id = 0
#     chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (new_frame_id, chunk_id))['arr_0']
#     chunk_cloud = open3d.geometry.PointCloud()
#     chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
#     chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])
#     address_zy = '/Users/zhuoyuelyu/Documents/a-Stanford/StanfordHCI/virtualhome-11/Assets/ply-common'
#     open3d.io.write_point_cloud('%s/%d.ply' % (address_zy, chunk_id), chunk_cloud)
#
#
#
# if __name__ == '__main__':
#     chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (new_frame_id, chunk_id))['arr_0']
#     chunk_cloud = open3d.geometry.PointCloud()
#     chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
#     chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])
#     address_zy = '/Users/zhuoyuelyu/Documents/a-Stanford/StanfordHCI/virtualhome-11/Assets/ply-common'
#     open3d.io.write_point_cloud('%s/%d.ply' % (address_zy, chunk_id), chunk_cloud)