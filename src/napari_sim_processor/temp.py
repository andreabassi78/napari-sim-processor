import napari
import numpy as np
import time
from skimage.draw import disk

N = 1000

def add_shapes(locations,radii):
    tic = time.time()
    shape_name='shapes'
    ellipses = []
    for center, radius in zip(locations, radii):
        bbox = np.array([center + np.array([radius, radius]),
                         center + np.array([radius, -radius]),
                         center + np.array([-radius, -radius]),
                         center + np.array([-radius, radius])]
                        )
        ellipses.append(bbox)
    if shape_name in viewer.layers: 
        circles_layer = viewer.layers[shape_name]
        circles_layer.data = ellipses 
    else:  
        circles_layer = viewer.add_shapes(name=shape_name,
                               edge_width = 3,
                               edge_color = 'red')
        circles_layer.add_ellipses(ellipses)
    print('time to create shapes:', time.time()-tic)   
      
def add_labels(locations,radii):
    tic = time.time()
    label_name='label'
    label_data = np.zeros([N,N], dtype=np.uint8)
    for center, radius in zip(locations, radii):
        rr, cc = disk(center, radius, shape=[N,N])
        label_data[rr, cc] =  1
    
    if label_name in viewer.layers: 
        circles_layer = viewer.layers[label_name]
        circles_layer.data = label_data
    else:  
        circles_layer = viewer.add_labels(label_data, name = label_name)
    print('time to create labels:', time.time()-tic) 
    
def add_circles(*args):
    locations = np.random.randint(200,N-200,[4,2])
    radii = np.random.randint(200,300,4)
    add_shapes(locations,radii) # takes 90ms on average
    add_labels(locations,radii) # takes 15ms on average

viewer = napari.Viewer()
viewer.add_image(np.random.random([10,N,N]))
viewer.dims.events.current_step.connect(add_circles)

napari.run()




