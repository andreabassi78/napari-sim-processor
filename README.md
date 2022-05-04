# napari-sim-processor

[![License](https://img.shields.io/pypi/l/napari-sim-processor.svg?color=green)](https://github.com/andreabassi78/napari-sim-processor/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sim-processor.svg?color=green)](https://pypi.org/project/napari-sim-processor)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sim-processor.svg?color=green)](https://python.org)
[![tests](https://github.com/andreabassi78/napari-sim-processor/workflows/tests/badge.svg)](https://github.com/andreabassi78/napari-sim-processor/actions)
[![codecov](https://codecov.io/gh/andreabassi78/napari-sim-processor/branch/main/graph/badge.svg)](https://codecov.io/gh/andreabassi78/napari-sim-processor)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sim-processor)](https://napari-hub.org/plugins/napari-sim-processor)

A Napari plugin for the reconstruction of Structured Illumination microscopy (SIM) data with GPU acceleration (with pytorch, if installed).
Currently supports:    
   - conventional data with improved resolution in 1D (1 angle, 3 phases)
   - conventional data (3 angles, 3 phases)
   - hexagonal SIM (1 angle, 7 phases).

The SIM processing widget accepts image stacks organized in 5D (angle,phase,z,y,x).

For raw image stacks with multiple z-frames each plane is processed as described in:
	https://doi.org/10.1098/rsta.2020.0162
    
A reshape widget is availbale, to easily reshape the data if they are organized differently than 5D (angle,phase,z,y,x)
    
Support for N angles, M phases is in progress.
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
