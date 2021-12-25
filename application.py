import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from app.config.settings import Settings


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    def __init__(self, width, height):

        self.settings = Settings()
        self.settings.width = width
        self.settings.height = height
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window("Interactive Segmentation of Point Clouds", width, height)
        w = self.window  # to make the code more concise

        # --------------------------------------------------------------------------------------------------------------
        # ---- Main panel ----------------------------------------------------------------------------------------------
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Materials", AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # event registration
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ---- Main panel ----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # ---- 3D scene representation ---------------------------------------------------------------------------------
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background(self.settings.scene_background)
        w.add_child(self._scene)

        # event registration
        self._scene.set_on_mouse(self._on_mouse)
        # ---- 3D scene representation ---------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # ---- Collapsible settings panel ------------------------------------------------------------------------------
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        material_settings = gui.CollapsableVert("Material settings", 0, gui.Margins(em, 0, 0, 0))
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        w.add_child(self._settings_panel)

        # event registration
        self._point_size.set_on_value_changed(self._on_point_size)
        w.set_on_layout(self._on_layout)
        # ---- Collapsible settings panel ------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self._apply_settings()

    def _apply_settings(self):
        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._point_size.double_value = self.settings.material.point_size

    # ---------------------------------------------------------------------
    # --- FIELD OF INTEREST
    # --- SHIFT + CLICK : pick a point from cloud to estimate distance map
    # ---------------------------------------------------------------------
    def _on_mouse(self, event: gui.MouseEvent):
        # projection of mouse click to scene (pick a point from point cloud)
        # 1 == LEFT_BUTTON, 4 == RIGHT_BUTTON (Open3D doesn't have python bindings for constants yet)
        if event.type == gui.MouseEvent.BUTTON_UP and event.buttons in [1, 4] \
                and event.is_modifier_down(gui.KeyModifier.SHIFT):

            # callback is being executed on secondary thread, but still blocking for now
            def depth_callback(depth_image):
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # get scene depth at clicked point
                depth = np.asarray(depth_image)[y, x]

                # clicked on nothing (empty scene)
                if depth == 1.0:
                    return

                # projected coordinates from screen to scene
                world = self._scene.scene.camera.unproject(
                    event.x, self.settings.height - event.y, depth, self._scene.frame.width, self._scene.frame.height
                )

                segm = Settings.SEGM
                if event.buttons == 4:
                    segm = Settings.NSEGM

                def update_geometry():
                    # add sphere at "picked point" position
                    self.settings.sphere[segm][Settings.ID] += 1
                    mat = rendering.Material()
                    mat.base_color = self.settings.sphere[segm][Settings.COLOR]
                    mat.shader = Settings.LIT
                    sphere = o3d.geometry.TriangleMesh.create_sphere(self.settings.sphere[Settings.SIZE])
                    sphere.compute_vertex_normals()
                    sphere.translate(world)
                    self._scene.scene.add_geometry(
                        self.settings.sphere[segm][Settings.NAME] + str(self.settings.sphere[segm][Settings.ID]),
                        sphere, mat
                    )

                    # compute distance map for interactive instance segmentation
                    if self.settings.point_cloud[Settings.DATA] is not None:
                        # kd tree from point cloud - optimisation purposes
                        legacy_pcl = self.settings.point_cloud[Settings.DATA].to_legacy_pointcloud()
                        kd_tree = o3d.geometry.KDTreeFlann(legacy_pcl)
                        # distance map
                        [_, idx, dist] = kd_tree.search_radius_vector_3d(
                            world, self.settings.distance_map[Settings.MAP_SIZE]
                        )

                        # TODO convert(legacy_pcl, world, idx, dist)

                        # update geometry (colors only)
                        np.asarray(legacy_pcl.colors)[idx[1:], :] = \
                            self.settings.distance_map[segm][Settings.COLOR]
                        self.settings.point_cloud[Settings.DATA] = \
                            o3d.t.geometry.PointCloud.from_legacy_pointcloud(legacy_pcl)
                        self._scene.scene.scene.update_geometry(
                            self.settings.point_cloud[Settings.ID],
                            self.settings.point_cloud[Settings.DATA],
                            rendering.Scene.UPDATE_COLORS_FLAG
                        )

                # update scene in main thread
                gui.Application.instance.post_to_main_thread(self.window, update_geometry)

            # get projected position from screen to scene from depth image rendering callback
            self._scene.scene.scene.render_to_depth_image(depth_callback)
            # returned event handled indicator
            return gui.Widget.EventCallbackResult.HANDLED

        # returned event ignored indicator
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_layout(self, theme):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * theme.font_size
        height = min(r.height, self._settings_panel.calc_preferred_size(theme).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load", self.window.theme)
        dlg.add_filter(".xyzrgb", "Point cloud files (.xyzrgb)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Interactive segmentation of point clouds\n"))
        dlg_layout.add_child(gui.Label("Developers:"))
        dlg_layout.add_child(gui.Label("Denis Dovičic (xdovic01@stud.fit.vutbr.cz)"))
        dlg_layout.add_child(gui.Label("Martin Minárik (xminar31@stud.fit.vutbr.cz)"))
        dlg_layout.add_child(gui.Label("Jozef Méry (xmeryj00@stud.fit.vutbr.cz)"))

        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        # has to be loaded as legacy point cloud and convert it to dtype=Float32 (float) tensor point cloud,
        # open3d.t.io.read_point_cloud creates dtype=Float64 (double) tensor - weird functionality, it's not efficient
        legacy_pcl = o3d.io.read_point_cloud(path)
        # conversion to dtype=Float32 point cloud from legacy point cloud
        self.settings.point_cloud[Settings.DATA] = o3d.t.geometry.PointCloud.from_legacy_pointcloud(legacy_pcl)
        if self.settings.point_cloud[Settings.DATA] is not None:
            try:
                self._scene.scene.add_geometry(
                    self.settings.point_cloud[Settings.ID],
                    self.settings.point_cloud[Settings.DATA],
                    self.settings.material
                )

                # set camera to look at center of point cloud
                self._scene.setup_camera(
                    60,
                    o3d.geometry.AxisAlignedBoundingBox(
                        self.settings.point_cloud[Settings.DATA].get_min_bound().numpy(),
                        self.settings.point_cloud[Settings.DATA].get_max_bound().numpy()
                    ),
                    self.settings.point_cloud[Settings.DATA].get_center().numpy()
                )
            except Exception as e:
                print(e)

    def export_image(self, path):

        def on_image(image):
            img = image

            quality = 9
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)