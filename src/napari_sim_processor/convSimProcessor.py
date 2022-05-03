import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from baseSimProcessor import BaseSimProcessor

class ConvSimProcessor(BaseSimProcessor):
    '''
    Implements conventional SIM system with two beams, three angles and three phase steps
    '''
    def __init__(self):
        self._nsteps = 9
        self._nbands = 3
        self.usePhases = False  # can be overridden before calibration
        super().__init__()

    def _get_band_construction_matrix(self, phi=None):
        '''
        Calculate the matrix (shape (self._nsteps, 2 * self._nbands +1)) that constructs illumination patterns in the
        stack from the different carrier components.
        Column 0 is the 0th (DC) order;
        Columns 1:self._nbands+1 are the complex phases of the self._nbands different carrier components
        The remaining columns are the complex conjugate of columns 1:self._nbands+1.
        Where a band is not present in the illumination for certain images the corresponding matrix element should be set to zero
        :param phi: is the shape(self._nbands, self._nsteps) array of measured phase steps in the calibration image
        :return: is the shape(self._nsteps, 2 * self._nbands + 1) matrix that constructs the illumination from the
        carrier components
        '''
        if phi is None:
            phi = np.array([0, 2 * pi / 3, 4 * pi / 3])
            phase_matrix = np.zeros((9,7))
            phase_matrix[0:3,1] = phi
            phase_matrix[0:3,4] = -phi
            phase_matrix[3:6,2] = phi
            phase_matrix[3:6,5] = -phi
            phase_matrix[6:9,3] = phi
            phase_matrix[6:9,6] = -phi
        else:
            phase_matrix = np.append(np.append(np.zeros((self._nsteps, 1)), phi.T, axis = 1), -phi.T, axis=1)
        ret = np.complex64(np.exp(1j * phase_matrix))
        ret[3:9,(1,4)] = 0
        ret[0:3,(2,5)] = 0
        ret[6:9,(2,5)] = 0
        ret[0:6,(3,6)] = 0
        if self.debug > 1:
            print(ret)
            print(np.angle(ret))
            plt.figure()
            plt.title('Matrix')
            plt.imshow(np.abs(ret))
        return ret

