import open3d.visualization.gui as gui


class ZhuoyueWindow:
    def __init__(self):
        self.window = gui.Application.instance.create_window("Augmented Home Assistant", 400, 400)
        w = self.window  # for more concise code
        em = w.theme.font_size
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        # Add Lights
        collapse = gui.CollapsableVert("Lights", 0.33 * em, gui.Margins(em, 0, 0, 0))
        switch = gui.ToggleSwitch("L1")
        switch.set_on_clicked(self._on_switch)
        collapse.add_child(switch)

        switch_2 = gui.ToggleSwitch("L2")
        switch_2.set_on_clicked(self._on_switch)
        collapse.add_child(switch_2)

        # Add doors
        collapse2 = gui.CollapsableVert("Doors", 0.33 * em, gui.Margins(em, 0, 0, 0))
        switch = gui.ToggleSwitch("D1")
        switch.set_on_clicked(self._on_switch)
        collapse2.add_child(switch)

        switch_2 = gui.ToggleSwitch("D2")
        switch_2.set_on_clicked(self._on_switch)
        collapse2.add_child(switch_2)

        layout.add_child(collapse)
        layout.add_child(collapse2)
        w.add_child(layout)

    def _on_switch(self, is_on):
        if is_on:
            print("Blanche is on")
        else:
            print("Blanche is stupid")


def main():
    gui.Application.instance.initialize()
    w = ZhuoyueWindow()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
