import numpy as np
import open3d as o3d
import os
import threading
import time
CLOUD_NAME = "points"
def main():
    MultiWinApp().run()


class MultiWinApp:

    def __init__(self):
        self.is_done = False
        self.n_snapshots = 0
        self.cloud = None
        self.main_vis = None
        self.snapshot_pos = None

    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer( "Augmented Home Assistant")
        self.main_vis.add_action("N/A",
                                 self.on_snapshot)
        self.main_vis.set_on_close(self.on_main_window_closing)

        app.add_window(self.main_vis)
        self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        threading.Thread(target=self.update_thread).start()

        app.run()

    def on_snapshot(self):
        print()

    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.

        self.cloud = o3d.io.read_point_cloud( "./cloud_bin_0.pcd")
        bounds = self.cloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()

        def add_first_cloud():
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)
            self.main_vis.reset_camera_to_default()
            self.main_vis.setup_camera(60, bounds.get_center(),
                                       bounds.get_center() + [0, 0, -3],
                                       [0, -1, 0])

        o3d.visualization.gui.Application.instance.post_to_main_thread(
            self.main_vis, add_first_cloud)

        while not self.is_done:
            time.sleep(0.1)

            # Perturb the cloud with a random walk to simulate an actual read
            pts = np.asarray(self.cloud.points)
            magnitude = 0.005 * extent
            displacement = magnitude * (np.random.random_sample(pts.shape) -
                                        0.5)
            new_pts = pts + displacement
            self.cloud.points = o3d.utility.Vector3dVector(new_pts)

            def update_cloud():
                # Note: if the number of points is less than or equal to the
                #       number of points in the original object that was added,
                #       using self.scene.update_geometry() will be faster.
                #       Requires that the point cloud be a t.PointCloud.
                self.main_vis.remove_geometry(CLOUD_NAME)
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)

            if self.is_done:  # might have changed while sleeping
                break
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.main_vis, update_cloud)


if __name__ == "__main__":
    main()
