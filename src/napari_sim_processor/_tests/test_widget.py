from napari_sim_processor import SimAnalysis, reshape
import numpy as np

def test_sim_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer() 
    sim_widget = SimAnalysis(viewer)
  

def test_reshape_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((3,6,100, 100)))
    reshape_widget = reshape()
    reshape_widget(viewer, viewer.layers[0],'apzyx',3,3,2,100,100)
    # captured = capsys.readouterr()
    assert viewer.layers[0].data.shape[2] == 2