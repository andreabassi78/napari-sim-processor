# napari-sim-processor

[![License](https://img.shields.io/pypi/l/napari-sim-processor.svg?color=green)](https://github.com/andreabassi78/napari-sim-processor/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sim-processor.svg?color=green)](https://pypi.org/project/napari-sim-processor)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sim-processor.svg?color=green)](https://python.org)
[![tests](https://github.com/andreabassi78/napari-sim-processor/workflows/tests/badge.svg)](https://github.com/andreabassi78/napari-sim-processor/actions)
[![codecov](https://codecov.io/gh/andreabassi78/napari-sim-processor/branch/main/graph/badge.svg)](https://codecov.io/gh/andreabassi78/napari-sim-processor)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sim-processor)](https://napari-hub.org/plugins/napari-sim-processor)

A Napari plugin for the reconstruction of Structured Illumination Microscopy (SIM) with GPU acceleration (with pytorch, if installed).
Currently supports:    
   - conventional SIM data with a generic number of angle and phases (typically, 3 angles and 3 phases are used for resolution improvement in 2D)
   - hexagonal SIM data with 7 phases.

The SIM processing widget accepts image stacks organized in 5D (`angle`,`phase`,`z`,`y`,`x`).

The reshape widget can be used to easily reshape the data if they are not organized as 5D (angle,phase,z,y,x).
Currently only square images are supported (`x`=`y`)

For raw image stacks with multiple z-frames each plane is processed as described here:
	https://doi.org/10.1098/rsta.2020.0162
        
Support for 3D SIM with enhanced resolution in all directions is not yet available.
Multicolor reconstruction is not yet available.  

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

You can install `napari-sim-processor` via [pip]:

    pip install napari-sim-processor


To install latest development version :

    pip install git+https://github.com/andreabassi78/napari-sim-processor.git


## Usage

1) Open napari. 

2) Launch the reshape and sim-processor widgets.

3) Open your raw image stack (using the napari built-in or your own file openers).

![raw](https://github.com/andreabassi78/napari-sim-processor/raw/main/images/Picture1.png)

4) If your image is ordered as a 5D stack (angle, phase, z-frame, y, x) go to point 6. 

5) In the reshape widget, select the actual number of acquired angles, phases, and frames (red arrow) and press `Reshape Stack`.
 Note that the label axis of the viewer will be updated (green arrow).

![raw](https://github.com/andreabassi78/napari-sim-processor/raw/main/images/Picture1b.png)

6) In the sim-reconstruction widget press the Select image layer button. Note that the number of phases and angles will be updated (blue arrow). 

7) Choose the correct parameters of the SIM acquisition (`NA`, `pixelsize`, `M`, etc.) and processing parameters (`alpha`, `beta`, w, `eta`, `group`):
   - `w`: parameter of the Weiner filter.
   - `eta`: constant used for calibration. It should be slightly smaller than the carrier frequency (in pupil radius units).
   - `group`: for stacks with multiple z-frames, it is the number of frames that are used together for the calibration process.
	
For details on the other parameters see https://doi.org/10.1098/rsta.2020.0162.

8) Calibrate the SIM processor, pressing the `Calibrate` button. This will find the carrier frequencies (red circles if the `Show Carrier` checkbox is selected), the modulation amplitude and the phase, using cross correlation analysis.

9) Click on the checkboxes to show the power spectrum of the raw image (`Show power spectrum`) or the cross-correlation (`Show Xcorr`), to see if the found carrier frequency is correct.

![raw](https://github.com/andreabassi78/napari-sim-processor/raw/main/images/Picture2b.png)
**Napari viewer showing the power spectrum of the raw stack. The pupil circle is in blue. A circle corresponding to `eta` is shown in green.**

![raw](https://github.com/andreabassi78/napari-sim-processor/raw/main/images/Picture2.png)
**Napari viewer showing the cross-correlation of the raw stack. The red circles indicate the found carrier frequencies**

10) Run the reconstruction of a single plane (`SIM reconstruction`) or of a stack (`Stack reconstruction`). After execution, a new image_layer will be added to the napari viewer. Click on the `Batch reconstruction` checkbox in order to process an entire stack in one shot. Click on the pytorch checkbox for gpu acceleration.

![raw](https://github.com/andreabassi78/napari-sim-processor/raw/main/images/Picture3b.png)
**Napari viewer with widgets showing a pseudo-widefield reconstruction**

![raw](https://github.com/andreabassi78/napari-sim-processor/raw/main/images/Picture3.png)
**Napari viewer with widgets showing a SIM reconstruction**

## GPU processing

The underlying processing classes will use numpy (and FFTW if available) for 
its calculations. For GPU accelerated processing you need to have either the 
PyTorch (tested with torch v1.11.0+cu113) or the CuPy (tested with cupy-cuda113 
v10.4.0) package installed.  Make sure to match the package cuda version to the CUDA library 
installed on your system otherwise PyTorch will default to CPU and CuPy will not work at all.  

Both packages give significant speedup on even relatively modest CUDA GPUs compared 
to Numpy, and PyTorch running on the CPU only can show improvements relative to numpy 
and FFTW. Selection of which processing package to use is via a ComboBox in the 
napari_sim_processor widget.  Only available packages are shown. 

Other than requiring a CUDA GPU it is advisable to have significant GPU memory 
available, particularly when processing large datasets.  Batch processing is the 
most memory hungry of the methods, but can process 280x512x512 datasets on a 4GB GPU.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-sim-processor" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/andreabassi78/napari-sim-processor/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
