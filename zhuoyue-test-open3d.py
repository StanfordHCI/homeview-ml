import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# class ZhuoyueWindow:

def zhuoyue_on_switch(is_on):
    if is_on:
        print("Onnnn")
    else:
        print("Offff")





if __name__ == '__main__':


    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Zhuoyue-Zhuoyue", 1024, 768)
    em = w.theme.font_size
    layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                     0.5 * em))
    collapse = gui.CollapsableVert("Widgets", 0.33 * em,
                                   gui.Margins(em, 0, 0, 0))
    switch = gui.ToggleSwitch("L1")
    switch.set_on_clicked(zhuoyue_on_switch)
    collapse.add_child(switch)

    #
    # o3d.visualization.webrtc_server.enable_webrtc()
    # cube_red = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    # cube_red.compute_vertex_normals()
    # cube_red.paint_uniform_color((1.0, 0.0, 0.0))
    # o3d.visualization.draw(cube_red)

    app.run()