# import numpy as np
# import open3d
# from utils import get_cloud, get_chunks, get_coords_and_colors, merge_clouds, fn
# from train import Model
# from glob import glob
# import numpy as np
# import config
# import open3d
# import torch
# import time

# # # convert all npz to ply
# def zhuoyue_convert_point_cloud():
#     dataset_name = "vh.cameras"
#     for frame_id in range(100):
#         for chunk_id in range(267):
#             chunk_points = np.load(dataset_name + '/chunk/%d-%d.npz' % (new_frame_id, chunk_id))['arr_0']
#             chunk_cloud = open3d.geometry.PointCloud()
#             chunk_cloud.points = open3d.utility.Vector3dVector(chunk_points[:, :3])
#             chunk_cloud.colors = open3d.utility.Vector3dVector(chunk_points[:, 3:])
#             address_zy = '/Users/zhuoyuelyu/Documents/a-Stanford/StanfordHCI/virtualhome-11/Assets/ply-common'
#             open3d.io.write_point_cloud('%s/%d.ply' % (address_zy, chunk_id), chunk_cloud)
#
#
# if __name__ == '__main__':
#     zhuoyue_convert_point_cloud()

# if __name__ == "__main__":
#     open3d.visualization.webrtc_server.enable_webrtc()
#     cube_red = open3d.geometry.TriangleMesh.create_box(1, 2, 4)
#     cube_red.compute_vertex_normals()
#     cube_red.paint_uniform_color((1.0, 0.0, 0.0))
#     open3d.visualization.draw(cube_red)


# examples/python/visualization/interactive_visualization.py

# import numpy as np
# import copy
# import open3d as o3d
#
#
# def demo_crop_geometry():
#     print("Demo for manual geometry cropping")
#     print(
#         "1) Press 'Y' twice to align geometry with negative direction of y-axis"
#     )
#     print("2) Press 'K' to lock screen and to switch to selection mode")
#     print("3) Drag for rectangle selection,")
#     print("   or use ctrl + left click for polygon selection")
#     print("4) Press 'C' to get a selected geometry and to save it")
#     print("5) Press 'F' to switch to freeview mode")
#     pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
#     o3d.visualization.draw_geometries_with_editing([pcd])
#
#
# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])
#
#
# def pick_points(pcd):
#     print("")
#     print(
#         "1) Please pick at least three correspondences using [shift + left click]"
#     )
#     print("   Press [shift + right click] to undo point picking")
#     print("2) After picking points, press 'Q' to close the window")
#     vis = o3d.visualization.VisualizerWithEditing()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     vis.run()  # user picks points
#     vis.destroy_window()
#     print("")
#     return vis.get_picked_points()
#
#
# def demo_manual_registration():
#     print("Demo for manual ICP")
#     source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
#     target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_2.pcd")
#     print("Visualization of two point clouds before manual alignment")
#     draw_registration_result(source, target, np.identity(4))
#
#     # pick points from two point clouds and builds correspondences
#     picked_id_source = pick_points(source)
#     picked_id_target = pick_points(target)
#     assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
#     assert (len(picked_id_source) == len(picked_id_target))
#     corr = np.zeros((len(picked_id_source), 2))
#     corr[:, 0] = picked_id_source
#     corr[:, 1] = picked_id_target
#
#     # estimate rough transformation using correspondences
#     print("Compute a rough transform using the correspondences given by user")
#     p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     trans_init = p2p.compute_transformation(source, target,
#                                             o3d.utility.Vector2iVector(corr))
#
#     # point-to-point ICP for refinement
#     print("Perform point-to-point ICP refinement")
#     threshold = 0.03  # 3cm distance threshold
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source, target, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())
#     draw_registration_result(source, target, reg_p2p.transformation)
#     print("")
#
#
# if __name__ == "__main__":
#     demo_crop_geometry()
#     demo_manual_registration()



# Exam the states of a specific IoT
if __name__ == '__main__':
    import json
    my_src = '/Users/zhuoyuelyu/Downloads/output-425-20-cameras-json/'
    # for i in range(171):
    #     f = open("{}/{}.json".format(my_src, i),)
    #     data = json.load(f)
    #     for entry in data:
    #         if entry['id'] == 402:
    #             print(entry['state'])
    count = 0
    for i in range(1):
        f = open("{}/{}.json".format(my_src, i),)
        data = json.load(f)
        for entry in data:
            if entry['class_name'] == 'door' or entry['class_name'] == 'light':
                print(count)
                print(entry['id'])
                print()
                count +=1

