import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from .baseSimProcessor import BaseSimProcessor

class ConvSimProcessor(BaseSimProcessor):
    '''
    Implements conventional SIM system with two beams, multiple angles and multiple phase steps
    The default number of angles and phase steps is 3
    '''
    def __init__(self, angleSteps=3, phaseSteps=3):
        assert phaseSteps >= 3
        assert angleSteps > 0
        self._nsteps = angleSteps * phaseSteps
        self._nbands = angleSteps
        self.etafac = np.ones(self._nbands)
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
        phaseSteps = self._nsteps // self._nbands
        if phi is None:
            phi = np.arange(phaseSteps) * 2 * pi / phaseSteps
            phase_matrix = np.zeros((self._nsteps, self._nbands * 2 + 1))
            for i in range(0, self._nbands):
                phase_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i] = phi
                phase_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i + self._nbands] = -phi
        else:
            phase_matrix = np.append(np.append(np.zeros((self._nsteps, 1)), phi.T, axis = 1), -phi.T, axis=1)
        mask_matrix = np.zeros((self._nsteps, self._nbands * 2 + 1))
        mask_matrix[:,0] = 1
        for i in range(0, self._nbands):
            mask_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i] = 1
            mask_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i + self._nbands] = 1
        ret = np.complex64(mask_matrix * np.exp(1j * phase_matrix))

        if self.debug > 1:
            print(ret)
            print(np.angle(ret))
            plt.figure()
            plt.title('Matrix')
            plt.imshow(np.abs(ret))
        return ret

from .baseSimProcessor3D import BaseSimProcessor3D

class ConvSimProcessor3D(BaseSimProcessor3D):
    '''
    Implements conventional SIM system with three beams, multiple angles and multiple phase steps
    The default number of angles is 3 and phase steps is 5
    '''
    def __init__(self, angleSteps=3, phaseSteps=3):
        assert phaseSteps >= 5
        assert angleSteps > 0
        self._nsteps = angleSteps * phaseSteps
        self._nbands = angleSteps * 2  # each angle has a low frequency and a high frequency band
        self.usePhases = False  # can be overridden before calibration
        self.etafac = np.ones(self._nbands)
        for i in range(0, self._nbands, 2):
            self.etafac[i] = 0.5    # For each angle the first carrier frequency is at 1/2 that of the second
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
        phaseSteps = self._nsteps // (self._nbands // 2)
        if phi is None:
            phi = np.arange(phaseSteps) * 2 * pi / phaseSteps
            phase_matrix = np.zeros((self._nsteps, self._nbands * 2 + 1))
            for i in range(0, self._nbands // 2):
                phase_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2] = phi
                phase_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2 + 1] = 2 * phi
                phase_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2 + self._nbands] = -phi
                phase_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2 + self._nbands + 1] = -2 * phi
        else:
            phase_matrix = np.append(np.append(np.zeros((self._nsteps, 1)), phi.T, axis = 1), -phi.T, axis=1)
        mask_matrix = np.zeros((self._nsteps, self._nbands * 2 + 1))
        mask_matrix[:,0] = 1
        for i in range(0, self._nbands // 2):
            mask_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2] = 1
            mask_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2 + 1] = 1
            mask_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2 + self._nbands] = 1
            mask_matrix[i * phaseSteps:(i + 1) * phaseSteps, 1 + i * 2 + self._nbands + 1] = 1
        ret = np.complex64(mask_matrix * np.exp(1j * phase_matrix))

        if self.debug > 1:
            print(ret)
            print(np.angle(ret))
            plt.figure()
            plt.title('Matrix')
            plt.imshow(np.abs(ret))
        return ret

