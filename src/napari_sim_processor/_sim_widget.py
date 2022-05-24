"""
Created on Tue Jan 25 16:34:41 2022

@authors: Andrea Bassi @Polimi, Mark Neil @ImperialCollege
"""
from napari_sim_processor.widget_settings import Setting, Combo_box
from napari_sim_processor.baseSimProcessor import pytorch, cupy
from napari_sim_processor.hexSimProcessor import HexSimProcessor
from napari_sim_processor.convSimProcessor import ConvSimProcessor
# from napari_sim_processor.simProcessor import SimProcessor
import napari
from qtpy.QtWidgets import QVBoxLayout,QSplitter, QHBoxLayout, QWidget, QPushButton, QLineEdit
from qtpy.QtCore import Qt
from napari.layers import Image
import numpy as np
from napari.qt.threading import thread_worker
from magicgui.widgets import FunctionGui
from magicgui import magicgui, magic_factory
import warnings
import time
from superqt.utils import qthrottled
from enum  import Enum


class Sim_modes(Enum):
    SIM = 1
    HEXSIM = 2


class Accel(Enum):
    USE_NUMPY = 1
    USE_TORCH = 2
    USE_CUPY = 3 
    
    @classmethod
    def available(cls):
        '''
        Returns a new Enum with the available cpu/gpu accelerations 
        '''
        available_list = [cls.USE_NUMPY]
        if pytorch: available_list.append(cls.USE_TORCH),
        if cupy: available_list.append(cls.USE_CUPY) 
        available_members = {member.name:member.value for member in available_list}
        return Enum('AvailableAccel', available_members)
    

def reshape_init(reshape_widget: FunctionGui):
    @reshape_widget.input_image.changed.connect
    def _on_image_changed(input_image: Image):
        shape = input_image.data.shape
        ndim = len(shape)
        if ndim < 3:
            raise(ValueError('Please select a >3D image'))
        sz,sy,sx = shape[-3::]
        reshape_widget.z.value = sz
        reshape_widget.y.value = sy
        reshape_widget.x.value = sx
        sp = shape[-4]  if ndim >3 else 1
        reshape_widget.phases.value = sp
        sa = shape[-5] if ndim > 4 else 1 
        reshape_widget.angles.value = sa
        if ndim > 5:
            raise(ValueError('Stack dimension >5D not supported'))
            

@magic_factory(widget_init=reshape_init,
               call_button="Reshape stack",
               x={'max': 4096},
               y={'max': 4096},
               input_order={"choices": ['apzyx', 'pazyx', 'pzayx', 'azpyx', 'zapyx', 'zpayx']},
               )
def reshape(viewer: napari.Viewer,
            input_image: Image,
            input_order = 'apzyx',
            angles:int=1, phases:int=1,
            z:int=1, y:int=1, x:int=1
            ):
    '''
    Reshape image to a 5D stack (angle,phase,z,y,x).
    Parameters
    ----------
    viewer : napari.Viewer
        Current viewer
    input_image : Image
        Input image to reshape.
    input_order : str
        Order of the input image. 
        a (angles), p (phases),
        z (slices), y-x (pixels)
        The default is 'apzyx'.
    angles : int
        Number of angles in the reshaped image. 
    phases : int
        Number of phases in the reshaped image. 
    z : int
        Number of slices in the reshaped image. 
    y : int
        Number of y pixels in the reshaped image. 
    x : int
        Number of x pixels in the reshaped image. It is automatically found on image change.
    '''
    data = input_image.data
    if angles*phases*z*y*x != data.size:
        raise(ValueError('Input image cannot be reshaped to the given values'))
    else:
        if input_order == 'apzyx':
            rdata = data.reshape(angles, phases, z, y, x)
        elif input_order == 'pazyx':
            #rdata = np.swapaxes(data.reshape(phases, angles, z, y, x),0,1)
            rdata = np.moveaxis(data.reshape(phases, z, angles, y, x),[0,1],[1,0])
        elif input_order == 'pzayx':
            rdata = np.moveaxis(data.reshape(phases, z, angles, y, x),[0,1,2],[1,2,0])
        elif input_order == 'azpyx':
            rdata = np.moveaxis(data.reshape(angles, z, phases, y, x), [1, 2], [2, 1])
        elif input_order == 'zapyx':
            rdata = np.moveaxis(data.reshape(z, angles, phases, y, x), [0, 1, 2], [2, 0, 1])
        elif input_order == 'zpayx':
            rdata = np.moveaxis(data.reshape(z, phases, angles, y, x), [0, 2], [2, 0])
        else:
            raise(ValueError('Input stack order reshaping not implemented'))
        input_image.data = rdata
        viewer.dims.axis_labels = ["angle", "phase", "z", "y","x"]
        viewer.dims.set_point(axis=[0,1,2], value=[0,0,0]) #raises ValueError in napari versions <0.4.13 


class SimAnalysis(QWidget):
    '''
    A Napari plugin for the reconstruction of Structured Illumination microscopy (SIM) data with GPU acceleration (with pytorch, if installed).
    Currently supports:    
   - conventional data with improved resolution in 1D (1 angle, 3 phases)
   - conventional data with a generic number of angles and phases
   - hexagonal SIM (1 angle, 7 phases).
    Accepts image stacks organized in 5D (angle,phase,z,y,x).
    For stacks with multiple z-frames each plane is processed as described in:
	https://doi.org/10.1098/rsta.2020.0162
    '''

    name = 'SIM_Analysis'
    
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()
        self.setup_ui() # run setup_ui before instanciating the settings
        self.start_sim_processor()
        self.viewer.dims.events.current_step.connect(self.on_step_change)
        
        
    def setup_ui(self):     
        def add_section(_layout):
            splitter = QSplitter(Qt.Vertical)
            _layout.addWidget(splitter)
            #_layout.addWidget(QLabel(_title))
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        top_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()
        layout.addLayout(top_layout)
        layout.addLayout(bottom_layout)
        left_layout = QVBoxLayout()
        bottom_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()
        bottom_layout.addLayout(right_layout)
        # Fill top layout
        self.add_magic_function(self.select_layer, top_layout)
        # Fill bottom-left layout
        self.sim_mode = Combo_box(name = 'Mode', choices = Sim_modes, layout=left_layout,
                                  write_function = self.reset_processor)
        self.phases_number = Setting('phases', dtype=int, initial=7, layout=left_layout, 
                              write_function = self.reset_processor)
        self.angles_number = Setting('angles', dtype=int, initial=1, layout=left_layout, 
                              write_function = self.reset_processor)
        self.magnification = Setting('M', dtype=float, initial=60,  
                                      layout=left_layout, write_function = self.setReconstructor)
        self.NA = Setting('NA', dtype=float, initial=1.05, layout=left_layout, 
                                       write_function = self.setReconstructor)
        self.n = Setting(name ='n', dtype=float, initial=1.33,  spinbox_decimals=2,
                                      layout=left_layout, write_function = self.setReconstructor)
        self.wavelength = Setting('\u03BB', dtype=float, initial=0.570,
                                       layout=left_layout,  spinbox_decimals=2, unit = '\u03BCm',
                                       write_function = self.setReconstructor)
        self.pixelsize = Setting('pixel size', dtype=float, initial=6.50, layout=left_layout,
                                  spinbox_decimals=2, unit = '\u03BCm',
                                  write_function = self.setReconstructor)
        self.dz = Setting('dz', dtype=float, initial=0.55, layout=left_layout,
                                  spinbox_decimals=2, unit = '\u03BCm',
                                  write_function = self.rescaleZ)
        self.alpha = Setting('alpha', dtype=float, initial=0.5,  spinbox_decimals=2, 
                              layout=left_layout, write_function = self.setReconstructor)
        self.beta = Setting('beta', dtype=float, initial=0.980, spinbox_step=0.01, 
                             layout=left_layout,  spinbox_decimals=3,
                             write_function = self.setReconstructor)
        self.w = Setting('w', dtype=float, initial=0.2, layout=left_layout,
                              spinbox_decimals=2,
                              write_function = self.setReconstructor)
        self.eta = Setting('eta', dtype=float, initial=0.65,
                            layout=left_layout, spinbox_decimals=3, spinbox_step=0.01,
                            write_function = self.setReconstructor)
        self.group = Setting('group', dtype=int, initial=30, vmin=1,
                            layout=left_layout,
                            write_function = self.setReconstructor)
        self.use_phases = Setting('use phases', dtype=bool, initial=True, layout=left_layout,                         
                                   write_function = self.setReconstructor)
        self.find_carrier = Setting('find carrier', dtype=bool, initial=True,
                                     layout=left_layout, 
                                     write_function = self.setReconstructor) 
        # Fill bottom-right layout    
        self.carrier_idx = Setting('carrier index', dtype=int, initial=0,
                                   layout=right_layout, vmin = 0,
                                   write_function = self.show_functions
                                   )
        self.showXcorr = Setting('Show Xcorr', dtype=bool, initial=False,
                                     layout=right_layout,
                                     write_function = self.show_functions
                                     )
        self.showSpectrum = Setting('Show Spectrum', dtype=bool, initial=False,
                                     layout=right_layout,
                                     write_function = self.show_functions
                                     )
        self.showWiener = Setting('Show Wiener filter', dtype=bool, initial=False,
                                     layout=right_layout,
                                     write_function = self.show_functions
                                     )
        self.showEta = Setting('Show Eta circle', dtype=bool, initial=False,
                                     layout=right_layout,
                                     write_function = self.show_functions
                                     )
        self.showCarrier = Setting('Show Carrier', dtype=bool, initial=False,
                                     layout=right_layout,
                                     write_function = self.show_functions
                                     )
        self.keep_calibrating = Setting('Continuos Calibration', dtype=bool, initial=False,
                                     layout=right_layout, 
                                     write_function = self.setReconstructor)
        self.batch = Setting('Batch Reconstruction', dtype=bool, initial=False,
                                     layout=right_layout, 
                                     write_function = self.setReconstructor)
        # creates the cpu/gpu acceleration combobox with only the availbale ones
        self.proc = Combo_box(name = 'Accel', choices = Accel.available(),
                                  layout=right_layout,
                                  write_function = self.setReconstructor)    
        # buttons
        buttons_dict = {'Widefield': self.calculate_WF_image,
                        'Calibrate': self.calibration,
                        'Plot calibration phases':self.find_phaseshifts,
                        'SIM reconstruction': self.single_plane_reconstruction,
                        'Stack SIM reconstruction': self.stack_reconstruction,
                        'Stack demodulation': self.stack_demodulation,
                        }
        for button_name, call_function in buttons_dict.items():
            button = QPushButton(button_name)
            button.clicked.connect(call_function)
            right_layout.addWidget(button)
        self.messageBox = QLineEdit()
        layout.addWidget(self.messageBox, stretch=True)
        self.messageBox.setText('Messages')
    
        
    def add_magic_function(self, function, _layout):
        self.viewer.layers.events.inserted.connect(function.reset_choices)
        self.viewer.layers.events.removed.connect(function.reset_choices)
        _layout.addWidget(function.native)
        
    
    @magicgui(call_button='Select image layer')    
    def select_layer(self, image: Image):
        '''
        Selects a Image layer after chaking that it contains raw sim data organized
        as a 5D stack (angle,phase,z,y,x).
        Stores the name of the image in self.imageRaw_name, which is used frequently in the other methods.
        Parameters
        ----------
        image : napari.layers.Image
            The image layer to process, it contains the raw data 
        '''      
        if not isinstance(image, Image):
            return
        if hasattr(self,'imageRaw_name'):
            delattr(self,'imageRaw_name')
        data = image.data
        if data.ndim != 5:
            raise(KeyError('Please select a 5D(angle,phase,z,y,x) stack'))
        self.imageRaw_name = image.name
        sa,sp,sz,sy,sx = image.data.shape
        if not sy == sx:
            raise(KeyError('Non-square images are not supported'))
        self.angles_number.val = sa
        self.phases_number.val = sp
        self.viewer.dims.axis_labels = ["angle", "phase", "z", "y","x"]
        self.rescaleZ()
        self.center_stack(image)
        self.move_layer_to_top(image)
        self.reset_processor()
        print(f'Selected image layer: {image.name}')
    
    
    def reset_processor(self,*args):
        '''
        Reset the processor and starts it (stops it if existing). 
        Disables xcorr,spectrum and circle.
        ''' 
        self.start_sim_processor()
        self.showXcorr.val = False
        self.showWiener.val = False
        self.showSpectrum.val = False
        self.showEta.val = False
        self.showCarrier.val = False
        self.keep_calibrating.val = False
        
            
    def rescaleZ(self):
        '''
        changes the z-scale of all the Images in layer with shape >=3D
        '''
        self.zscaling = self.dz.val /(self.pixelsize.val/self.magnification.val)
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                if layer.ndim >=3:
                    scale = layer.scale 
                    scale[-3] = self.zscaling
                    layer.scale = scale
                    
      
    def center_stack(self, image_layer):
        '''
        centers a >3D stack in z,y,x 
        '''
        data = image_layer.data
        if data.ndim >2:
            current_step = list(self.viewer.dims.current_step)
            for dim_idx in [-3,-2,-1]:
                current_step[dim_idx] = data.shape[dim_idx]//2
            self.viewer.dims.current_step = current_step                
           
            
    @qthrottled (timeout=10)
    def on_step_change(self, *args):   
        if hasattr(self, 'imageRaw_name'):
            t0 = time.time()
            self.setReconstructor()
            min_timeout = 10 #ms
            delta_t = time.time() - t0
            self.on_step_change.set_timeout(delta_t + min_timeout)


    def show_image(self, image_values, im_name, **kwargs):
        '''
        creates a new Image layer with image_values as data
        or updates an existing layer, if 'hold' in kwargs is True 
        '''
        if 'scale' in kwargs.keys():    
            scale = kwargs['scale']
        else:
            scale = [1.]*image_values.ndim
        if 'colormap' in kwargs.keys():
            colormap = kwargs['colormap']
        else:
            colormap = 'gray'    
        if kwargs.get('hold') is True and im_name in self.viewer.layers:
            layer = self.viewer.layers[im_name]
            layer.data = image_values
            layer.scale = scale
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = im_name,
                                            scale = scale,
                                            colormap = colormap,
                                            interpolation = 'bilinear')
        self.center_stack(image_values)
        # self.move_layer_to_top(layer)
        if kwargs.get('autoscale') is True:
            layer.reset_contrast_limits()
        return layer


    def remove_layer(self, layer):
        if layer in self.viewer.layers:
            self.viewer.layers.remove(layer.name)    
    
    
    def move_layer_to_top(self, layer_to_move):
        '''
        Moves the layer to the top of the viewer and selects it
        '''
        for idx,layer in enumerate(self.viewer.layers):
            #couldn't find a way to get the index of a certain layer directly from the Layer object
            if layer is layer_to_move:
                self.viewer.layers.move(idx, len(self.viewer.layers))
                layer.visible = True
                if isinstance(layer, Image):
                    self.viewer.layers.selection = [layer]
    
    
    def make_layers_visible(self, *layers_list):
        '''
        Makes all the passed layers visible, while making all the others invisible
        '''
        for layer in self.viewer.layers:
            if layer in layers_list:
                layer.visible = True
            else:
                layer.visible = False    
    
    
    def is_image_in_layers(self):
        '''
        Checks if the raw image has been selected
        '''
        if hasattr(self, 'imageRaw_name'):
            if self.imageRaw_name in self.viewer.layers:
                return True
        return False   
    
    
    def get_hyperstack(self):
        '''
        Returns the full 5D (angles,phases,z,y,x) raw image stack
        '''
        try:
            return self.viewer.layers[self.imageRaw_name].data
        except:
            raise(KeyError('Please select a valid stack'))
    
    
    def get_current_ap_stack(self):
        '''
        Returns the 4D raw image (angles,phases,y,x) stack at the z value selected in the viewer  
        '''
        fullstack = self.get_hyperstack()
        z_index = int(self.viewer.dims.current_step[2])
        s = fullstack.shape
        assert z_index < s[-3], 'Please choose a valid z step for the selected stack'
        stack = fullstack[:,:,z_index,:,:]
        return stack
    
    
    def get_current_stack_for_calibration(self):
        '''
        Returns the 4D raw image (angles,phases,y,x) stack at the z value selected in the viewer  
        '''
        data = self.get_hyperstack()
        dshape = data.shape
        zidx = int(self.viewer.dims.current_step[2])
        delta = self.group.val // 2
        remainer = self.group.val % 2
        zmin = max(zidx-delta,0)
        zmax = min(zidx+delta+remainer,dshape[2])
        new_delta = zmax-zmin
        data = data[...,zmin:zmax,:,:]
        phases_angles = self.phases_number.val*self.angles_number.val
        rdata = data.reshape(phases_angles, new_delta, dshape[-2],dshape[-1])            
        cal_stack = np.swapaxes(rdata, 0, 1).reshape((phases_angles * new_delta, dshape[-2],dshape[-1]))
        return cal_stack
        
    
    def get_current_image(self):
        '''
        Returns the 2D raw image stack at the z, angle and phase values selected in the viewer  
        '''
        hs = self.get_hyperstack()
        z_index = int(self.viewer.dims.current_step[2])
        phase_index = int(self.viewer.dims.current_step[1])
        angle_index = int(self.viewer.dims.current_step[0])   
        img0 = hs[angle_index,phase_index,z_index,:,:]
        return(img0)
    
    
    def start_sim_processor(self):
        ''''
        Creates an instance of the Processor
        '''
        if self.is_image_in_layers():
            self.isCalibrated = False
            if hasattr(self, 'h'):
                self.stop_sim_processor()
                self.start_sim_processor()
            else:
                if self.sim_mode.current_data == Sim_modes.HEXSIM.value:  
                    self.h = HexSimProcessor()  
                    k_shape = (3,1)
                elif self.sim_mode.current_data == Sim_modes.SIM.value and self.phases_number.val >= 3 and self.angles_number.val > 0:
                    self.h = ConvSimProcessor(angleSteps=self.angles_number.val,
                                              phaseSteps=self.phases_number.val)
                    k_shape = (self.angles_number.val,1)   
                else: 
                    raise(ValueError("Invalid phases or angles number"))
                self.carrier_idx.set_min_max(0,k_shape[0]-1) # TODO connect carrier idx to angle if Sim_mode == SIM
                self.h.debug = False
                self.setReconstructor() 
                self.kx_input = np.zeros(k_shape, dtype=np.single)
                self.ky_input = np.zeros(k_shape, dtype=np.single)
                self.p_input = np.zeros(k_shape, dtype=np.single)
                self.ampl_input = np.zeros(k_shape, dtype=np.single)

            
    def stop_sim_processor(self):
        if hasattr(self, 'h'):
            delattr(self, 'h')

    
    def setReconstructor(self,*args):
        '''
        Sets the attributes of the Processor
        Executed frequently, upon update of several settings
        '''
        if hasattr(self, 'h'):   
            self.h.usePhases = self.use_phases.val
            self.h.magnification = self.magnification.val
            self.h.NA = self.NA.val
            self.h.n = self.n.val
            self.h.wavelength = self.wavelength.val
            self.h.pixelsize = self.pixelsize.val
            self.h.alpha = self.alpha.val
            self.h.beta = self.beta.val
            self.h.w = self.w.val
            self.h.eta = self.eta.val
            if not self.find_carrier.val:
                self.h.kx = self.kx_input
                self.h.ky = self.ky_input
            if self.keep_calibrating.val:
                self.calibration()
            self.show_functions()
          
           
    def show_wiener(self, *args):
        """
        Shows the Wiener filter 
        """
        if self.is_image_in_layers():
            imname = 'Wiener_' + self.imageRaw_name
            if self.isCalibrated and self.showWiener.val:
                
                img = self.h.wienerfilter
                swy,swx = img.shape
                self.show_image(img[swy//2-swy//4:swy//2+swy//4,swx//2-swx//4:swx//2+swx//4],
                                imname, hold = True, scale=[1,1])
            elif not self.showWiener.val and imname in self.viewer.layers:
                self.remove_layer(self.viewer.layers[imname])
                       
       
    def show_spectrum(self, *args):
        """
        Calculates and shows the power spectrum of the image
        """
        from numpy.fft import fft2, fftshift
        if self.is_image_in_layers():
            imname = 'Spectrum_' + self.imageRaw_name
            if self.showSpectrum.val:
                img0 = self.get_current_image()
                epsilon = 1e-10
                ps = np.log((np.abs(fftshift(fft2(img0))))**2+epsilon)
                self.show_image(ps, imname, hold = True)
            elif not self.showSpectrum.val and imname in self.viewer.layers:
                self.remove_layer(self.viewer.layers[imname])
       
    
    def show_xcorr(self, *args):
        """
        Show the crosscorrelation of the low and high pass filtered version of the raw images,
        used forfinding the carrier
        """
        if self.is_image_in_layers():
            imname = 'Xcorr_' + self.imageRaw_name
            if self.showXcorr.val and hasattr(self,'h'):
                im = self.get_current_stack_for_calibration()
                # choose the gpu acceleration
                if self.proc.current_data == Accel.USE_TORCH.value:
                    ixf = np.squeeze(self.h.crossCorrelations_pytorch(im)) #TODO check if traspose or flip should be added
                elif self.proc.current_data == Accel.USE_CUPY.value:
                    ixf = np.squeeze(self.h.crossCorrelations_cupy(im))
                else:
                    ixf = np.squeeze(self.h.crossCorrelations(im))
                # show the selected carrier
                if ixf.ndim >2:
                    carrier_idx = self.carrier_idx.val
                    ixf =ixf [carrier_idx,:,:]   
                else:
                    self.carrier_idx.set_min_max(0,0)
                self.show_image(ixf, imname, hold = True,
                                colormap ='twilight', autoscale = True)
            elif not self.showXcorr.val and imname in self.viewer.layers:
                self.remove_layer(self.viewer.layers[imname])
    
    def show_functions(self, *args):
        self.show_wiener()
        self.show_spectrum()
        self.show_xcorr()
        self.show_carrier()
        self.show_eta()
        
     
    def calculate_kr(self,N):  
        '''
        Parameter:
            N: number of pixels of the image
        Returns: 
            cutoff: pupil cutoff frequency in pixels number
            dk: sampling in spatial frequancy domain
        '''
        dx = self.h.pixelsize / self.h.magnification  # Sampling in image plane
        res = self.h.wavelength / (2 * self.h.NA)
        cutoff = 1 # coherent cutoff frequency
        oversampling = res / dx
        dk = oversampling / (N / 2) # k space is normalised to coherent cutoff
        cutoff_in_pixels = cutoff / dk
        return cutoff_in_pixels, dk   
      
     
    def show_carrier(self, *args):
        '''
        Draws the carrier frenquencies in a shape layer
        '''
        if self.is_image_in_layers() :
            name = f'carrier_{self.imageRaw_name}'
            if self.showCarrier.val and self.isCalibrated:
                N = self.h.N
                cutoff, dk = self.calculate_kr(N)
                kxs = self.h.kx
                kys = self.h.ky
                pc = np.zeros((len(kxs),2))
                radii = []
                for idx, (kx,ky) in enumerate(zip(kxs,kys)):
                    pc[idx,0] = ky[0] / dk + N/2
                    pc[idx,1] = kx[0] / dk + N/2
                    radii.append(self.h.N // 30) # radius of the displayed circle
                layer=self.add_circles(pc, radii, name, color='red')
                self.move_layer_to_top(layer) 
                # kr = np.sqrt(kxs**2+kys**2)
                # print('Carrier magnitude / cut off:', *kr/cutoff*dk)
            elif name in self.viewer.layers:
                self.remove_layer(self.viewer.layers[name])
    
                   
    def show_eta(self):
        '''
        Shows two circles with radius eta (green circle), 
        and with the radius of the pupil (blue), NA/lambda 
        '''
        if self.is_image_in_layers():
            name = f'eta_circle_{self.imageRaw_name}'
            centres = []
            radii = []
            colors = []
            if self.showEta.val:
                N = self.h.N
                cutoff, dk   = self.calculate_kr(N)  
                eta_radius = 1.9 * self.h.eta * cutoff
                centres.append(np.array([N/2,N/2]))
                radii.append(eta_radius)
                colors.append('green')
                centres.append(np.array([N/2,N/2]))
                radii.append(2 * cutoff)
                colors.append('blue')
                layer=self.add_circles(centres, radii, shape_name=name, color=colors)
                self.move_layer_to_top(layer)
            elif name in self.viewer.layers:
                self.remove_layer(self.viewer.layers[name])   

    
    def add_circles(self, locations, radii,
                    shape_name='shapename', color='blue', hold=False):
        '''
        Creates a circle in a layer with yx coordinates speciefied in each row of locations
        
        Parameters
        ----------
        locations : np.array
            yx coordinates of the centers. 
        shape_name : str
            name of the new Shape
        radii : int
            radii of the circles.
        color : list of or single instance of str or RGBA list
            color of the circles.
        hold : bool
            if True updates the existing layer, with name shape_name,
            without creating a new layer
        '''
        ellipses = []
        for center, radius in zip(locations, radii):
            bbox = np.array([center + np.array([radius, radius]),
                             center + np.array([radius, -radius]),
                             center + np.array([-radius, -radius]),
                             center + np.array([-radius, radius])]
                            )
            ellipses.append(bbox)
        
        if shape_name in self.viewer.layers: 
            circles_layer = self.viewer.layers[shape_name]
            if hold:
                circles_layer.add_ellipses(ellipses, edge_color=color)
            else:
                circles_layer.data = ellipses
        else:  
            circles_layer = self.viewer.add_shapes(name=shape_name,
                                   edge_width = 1.3,
                                   face_color = [1, 1, 1, 0])
            circles_layer.add_ellipses(ellipses, edge_color=color)
        return circles_layer
    
    
    def showCalibrationTable(self):
        import pandas as pd
        headers= ['kx_in','ky_in','kx','ky','phase','amplitude']
        vals = [self.kx_input, self.ky_input,
                self.h.kx, self.h.ky,
                self.h.p,self.h.ampl]
        table = pd.DataFrame([vals] , columns = headers )
        print(table)
            
        
    def stack_demodulation(self, *args):
        '''
        Demodulates the data as proposed in Neil et al, Optics Letters 1997.
        '''
        assert self.isCalibrated, 'SIM processor not calibrated'    
        
        fullstack = self.get_hyperstack()
        sa,sp,sz,sy,sx = fullstack.shape
        phases_angles = sa*sp
        pa_stack = fullstack.reshape(phases_angles, sz, sy, sx)
        paz_stack = np.swapaxes(pa_stack, 0, 1).reshape((phases_angles*sz, sy, sx))
        if self.proc.current_data == Accel.USE_TORCH.value:
            demodulation_function = self.h.filteredOSreconstruct_pytorch
        elif self.proc.current_data == Accel.USE_CUPY.value:
            demodulation_function = self.h.filteredOSreconstruct_cupy
        else:
            demodulation_function = self.h.filteredOSreconstruct
        demodulated = demodulation_function(paz_stack)
        if demodulated.ndim < 3:
            demodulated = demodulated[np.newaxis, :]
        imname = 'Demodulated_' + self.imageRaw_name
        scale = [self.zscaling,1,1]
        self.show_image(demodulated, imname, scale=scale, hold=True, autoscale=True)
        #print('Stack demodulation completed')
        
    
    def calculate_WF_image(self):
        '''
        Calculates and shows the widefield image from the raw 5D image stack.
        It averages the data on all phases and angles.
        Shows the resulting 3D image stack as an Image layer of the viewer.
        '''
        imageWFdata = np.mean(self.get_hyperstack(), axis=(0,1))
        imname = 'WF_' + self.imageRaw_name
        scale = self.viewer.layers[self.imageRaw_name].scale
        self.show_image(imageWFdata, imname, scale = scale[2:], hold = True, autoscale = True)
        
    
    def calibration(self, *args):
        '''
        Performs the data calibration using the Processor (self.h).
        It is performed on a stack of images around the frame selected in the viewer.
        The size of the stack is the value specified in the "group" Setting.
        *args is to avoid conflic with the add_timer decorator
        '''
        if hasattr(self, 'h'):
            imRaw = self.get_current_stack_for_calibration()
            start_time = time.time()
            if self.proc.current_data == Accel.USE_TORCH.value:
                self.h.calibrate_pytorch(imRaw,self.find_carrier.val)
            elif self.proc.current_data == Accel.USE_CUPY.value:
                self.h.calibrate_cupy(imRaw, self.find_carrier.val)
            else:
                self.h.calibrate(imRaw,self.find_carrier.val)
            elapsed_time = time.time() - start_time
            self.messageBox.setText(f'Calibration time {elapsed_time:.3f}s')
            self.isCalibrated = True
            if self.find_carrier.val: # store the value found   
                self.kx_input = self.h.kx  
                self.ky_input = self.h.ky
                self.p_input = self.h.p
                self.ampl_input = self.h.ampl 
            self.show_functions()
            
                   
    def single_plane_reconstruction(self):
        '''
        Performs SIM reconstruction on the selected z plane.
        '''
        assert self.isCalibrated, 'SIM processor not calibrated, unable to perform SIM reconstruction'
        current_image = self.get_current_ap_stack()
        dshape= current_image.shape
        phases_angles = self.phases_number.val*self.angles_number.val
        rdata = current_image.reshape(phases_angles, dshape[-2],dshape[-1])
        if self.proc.current_data == Accel.USE_TORCH.value:
            imageSIM = self.h.reconstruct_pytorch(rdata.astype(np.float32)) #TODO:this is left after conversion from torch
        elif self.proc.current_data == Accel.USE_CUPY.value:
            imageSIM = self.h.reconstruct_cupy(rdata.astype(np.float32))  # TODO:this is left after conversion from torch
        else:
            imageSIM = self.h.reconstruct_rfftw(rdata)
        imname = 'SIM_' + self.imageRaw_name
        self.show_image(imageSIM, im_name=imname, scale=[0.5,0.5], hold =True, autoscale = True)
    
    
    def stack_reconstruction(self):
        '''
        Performs SIM reconstruction on entire data (5D raw image stack).
        Performs plane-by-plane reconstruction (_stack_reconstruction), 
            calibrating the Processor continuosly if "Continuous calibration" checkbox is selected
        Performs batch reconstruction if "Batch reconstrction" checkbox is selected
        '''
        def update_sim_image(stack):
            imname = 'SIMstack_' + self.imageRaw_name
            scale = [self.zscaling, 0.5, 0.5]
            self.show_image(stack, im_name=imname, scale=scale, hold = True, autoscale = True)
        
        @thread_worker(connect={'returned': update_sim_image})
        def _stack_reconstruction():
            warnings.filterwarnings('ignore')
            stackSIM = np.zeros([sz,2*sy,2*sx], dtype=np.single)
            for zidx in range(sz):
                phases_stack = np.squeeze(pa_stack[:,zidx,:,:])
                if self.keep_calibrating.val:
                    delta = self.group.val // 2
                    if (delta == 0) or (zidx % delta == 0):
                        remainer = self.group.val % 2
                        zmin = max(zidx-delta,0)
                        zmax = min(zidx+delta+remainer,sz)
                        new_delta = zmax-zmin
                        data = pa_stack[:,zmin:zmax,:,:]
                        s_pa = data.shape[0]
                        selected_imRaw = np.swapaxes(data, 0, 1).reshape((s_pa * new_delta, sy, sx))
                        if self.proc.current_data == Accel.USE_TORCH.value:
                            self.h.calibrate_pytorch(selected_imRaw,self.find_carrier.val)
                        elif self.proc.current_data == Accel.USE_CUPY.value:
                            self.h.calibrate_cupy(selected_imRaw, self.find_carrier.val)
                        else:
                            self.h.calibrate(selected_imRaw,self.find_carrier.val)                
                if self.proc.current_data == Accel.USE_TORCH.value:
                    stackSIM[zidx,:,:] = self.h.reconstruct_pytorch(phases_stack.astype(np.float32)) #TODO:this is left after conversion from torch
                elif self.proc.current_data == Accel.USE_CUPY.value:
                    stackSIM[zidx, :, :] = self.h.reconstruct_cupy(phases_stack.astype(np.float32))  # TODO:this is left after conversion from torch
                else:
                    stackSIM[zidx,:,:] = self.h.reconstruct_rfftw(phases_stack)      
            return stackSIM
        
        @thread_worker(connect={'returned': update_sim_image})
        def _batch_reconstruction():
            warnings.filterwarnings('ignore')
            start_time = time.time()
            if self.proc.current_data == Accel.USE_TORCH.value:
                stackSIM = self.h.batchreconstructcompact_pytorch(paz_stack, blocksize = 32)
            elif self.proc.current_data == Accel.USE_CUPY.value:
                stackSIM = self.h.batchreconstructcompact_cupy(paz_stack, blocksize=32)
            else:
                stackSIM = self.h.batchreconstructcompact(paz_stack)
            elapsed_time = time.time() - start_time
            self.messageBox.setText(f'Batch reconstruction time {elapsed_time:.3f}s')
            return stackSIM
        
        # main function executed here
        assert self.isCalibrated, 'SIM processor not calibrated, unable to perform SIM reconstruction'
        fullstack = self.get_hyperstack()
        sa,sp,sz,sy,sx = fullstack.shape
        phases_angles = sa*sp
        pa_stack = fullstack.reshape(phases_angles, sz, sy, sx)
        paz_stack = np.swapaxes(pa_stack, 0, 1).reshape((phases_angles*sz, sy, sx))
        if self.batch.val:
            _batch_reconstruction()
        else: 
            _stack_reconstruction()
                
          
    def find_phaseshifts(self):
        assert self.isCalibrated, 'SIM processor not calibrated, unable to show phases'
        if self.sim_mode.current_data == Sim_modes.HEXSIM.value:
            self.find_hexsim_phaseshifts()
        elif self.sim_mode.current_data == Sim_modes.SIM.value:
            self.find_sim_phaseshifts()
        
        
    def find_hexsim_phaseshifts(self):   
        phaseshift = np.zeros((7,3))
        expected_phase = np.zeros((7,3))
        error = np.zeros((7,3))
        stack = self.get_current_ap_stack()
        sa,sp,sy,sx = stack.shape
        img = stack.reshape(sa*sp, sy, sx) 
        for i in range (3):
            if self.proc.current_data == Accel.USE_TORCH.value:
                phase, _ = self.h.find_phase_pytorch(self.h.kx[i], self.h.ky[i], img)
            elif self.proc.current_data == Accel.USE_CUPY.value:
                phase, _ = self.h.find_phase_cupy(self.h.kx[i], self.h.ky[i], img)
            else:
                phase, _ = self.h.find_phase(self.h.kx[i], self.h.ky[i], img)
            expected_phase[:,i] = np.arange(7) * 2*(i+1) * np.pi / 7
            phaseshift[:,i] = np.unwrap(phase - phase[0] - expected_phase[:,i]) + expected_phase[:,i]
        error = phaseshift-expected_phase
        data_to_plot = [expected_phase, phaseshift, error]
        symbols = ['.','o','|']
        legend = ['expected', 'measured', 'error']
        self.plot_with_plt(data_to_plot, legend, symbols,
                                xlabel = 'step', ylabel = 'phase (rad)', vmax = 6*np.pi)
            
    
    def find_sim_phaseshifts(self):   
        stack = self.get_current_ap_stack()
        sa,sp,sy,sx = stack.shape
        img = stack.reshape(sa*sp, sy, sx)  
        for angle_idx in range (sa):
            phaseshift = np.zeros((sp,sa))
            expected_phase = np.zeros((sp,sa))
            error = np.zeros((sp,sa))
            if self.proc.current_data == Accel.USE_TORCH.value:
                phase, _ = self.h.find_phase_pytorch(self.h.kx[angle_idx], self.h.ky[angle_idx], img)
            elif self.proc.current_data == Accel.USE_CUPY.value:
                phase, _ = self.h.find_phase_cupy(self.h.kx[angle_idx], self.h.ky[angle_idx], img)
            else:
                phase, _ = self.h.find_phase(self.h.kx[angle_idx], self.h.ky[angle_idx], img)
            phase = np.unwrap(phase)
            phase = phase.reshape(sa,sp).T
            expected_phase[:,angle_idx] = np.arange(sp) * 2*np.pi / sp
            phaseshift= phase-phase[0,:]
            error = phaseshift-expected_phase      
            data_to_plot = [expected_phase[:,angle_idx], phaseshift[:,angle_idx], error[:,angle_idx]]
            symbols = ['.','o','|']
            legend = ['expected', 'measured', 'error']
            self.plot_with_plt(data_to_plot, legend, symbols, title = f'angle {angle_idx}',
                                    xlabel = 'step', ylabel = 'phase (rad)', vmax = 2*np.pi)
                             
    
    def plot_with_plt(self, data_list, legend, symbols,
                      xlabel = 'step', ylabel = 'phase',
                      vmax = 2*np.pi, title = ''):
        import matplotlib.pyplot as plt
        char_size = 10
        plt.rc('font', family='calibri', size=char_size)
        fig = plt.figure(figsize=(4,3), dpi=150)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(xlabel, size=char_size)
        ax.set_ylabel(ylabel, size=char_size)
        ax.set_title(title, size=char_size)
        s = data_list[0].shape
        cols = 1 if len(s)==1 else s[1]
        colors = ('black','red','green')
        for cidx in range(cols):
            color = colors[cidx%3]
            for idx, data in enumerate(data_list):
                column = data if len(s)==1 else data[...,cidx]
                marker = symbols[idx]
                linewidth = 0.2 if marker == 'o' else 0.8
                ax.plot(column, marker=marker, linewidth =linewidth, color=color)    
       
        ax.xaxis.set_tick_params(labelsize=char_size*0.75)
        ax.yaxis.set_tick_params(labelsize=char_size*0.75)
        ax.legend(legend, loc='best', frameon = False, fontsize=char_size*0.8)
        ax.grid(True, which='major', axis='both', alpha=0.2)
        vales_num = s[0]
        ticks = np.linspace(0, vmax*(vales_num-1)/vales_num, 2*vales_num-1 )
        ax.set_yticks(ticks)
        fig.tight_layout()
        plt.show(block=False)
        plt.rcParams.update(plt.rcParamsDefault)


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
     