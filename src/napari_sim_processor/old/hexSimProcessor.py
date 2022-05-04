import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from baseSimProcessor import BaseSimProcessor

class HexSimProcessor(BaseSimProcessor):
    '''
    Implements hexagonal SIM illumination with three beams, seven phase steps
    '''
    def __init__(self):
        self._nsteps = 7
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
            phase_matrix = (2 * pi / self._nsteps) * (
                        (np.arange(0, self._nsteps)[:, np.newaxis]) * np.arange(0, self._nbands + 1))
            phase_matrix = np.append(phase_matrix, -phase_matrix[:, 1:], axis=1)
        else:
            phase_matrix = np.append(np.append(np.zeros((self._nsteps, 1)), phi.T, axis = 1), -phi.T, axis=1)
        if self.debug:
            print(phase_matrix)
        M = np.complex64(np.exp(1j * phase_matrix))
        if self.debug:
            print(np.linalg.cond(M), np.linalg.cond(M, 'fro'), np.linalg.cond(M, np.inf), np.linalg.cond(M, -np.inf),
                  np.linalg.cond(M, 1), np.linalg.cond(M, -1), np.linalg.cond(M, 2), np.linalg.cond(M, -2))
        return M

