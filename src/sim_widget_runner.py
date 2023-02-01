from napari_sim_processor._sim_widget import reshape, SimAnalysis
'''
Script that runs the napari widget from the IDE. 
It is not executed when the plugin runs.
'''

if __name__ == '__main__':

    import napari
    viewer = napari.Viewer()
    widget = SimAnalysis(viewer)
    my_reshape_widget = reshape()    
    viewer.window.add_dock_widget(my_reshape_widget, name = 'Reshape stack', add_vertical_stretch = True)
    viewer.window.add_dock_widget(widget,
                                  name = 'Sim analyzer @Polimi',
                                  add_vertical_stretch = True)
    napari.run() 
     