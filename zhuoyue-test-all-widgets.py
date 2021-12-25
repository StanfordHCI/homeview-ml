import open3d.visualization.gui as gui


class ZhuoyueWindow:
    def __init__(self):
        self.window = gui.Application.instance.create_window("Augmented Home Assistant", 400, 400)
        w = self.window  # for more concise code
        em = w.theme.font_size
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        # Add Lights
        self.iots = gui.CollapsableVert("IoTs", 0.33 * em, gui.Margins(em, 0, 0, 0))
        num_lights = 5
        self.iots.add_child(gui.Label("Lights"))
        for i in range(num_lights):
            self.iots.add_child(self.add_iot("L" + str(i)))

        num_doors = 3
        self.iots.add_child(gui.Label("Doors"))
        for i in range(num_doors):
            self.iots.add_child(self.add_iot("D" + str(i)))

        layout.add_child(self.iots)
        w.add_child(layout)

    def add_iot(self, name):
        switch = gui.ToggleSwitch(name)
        switch.set_on_clicked(self.on_switch)
        return switch

    def on_switch(self, is_on):
        # get the states of all toggles
        iot_states = [int(x.is_on) for x in self.iots.get_children() if type(x).__name__ == "ToggleSwitch"]
        print(iot_states)



def main():
    gui.Application.instance.initialize()
    w = ZhuoyueWindow()
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
