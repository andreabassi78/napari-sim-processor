import multiprocessing
import numpy as np
import scipy.ndimage
import scipy.io
from numpy import exp, pi, sqrt, log2, arccos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .baseSimProcessor import BaseSimProcessor

try:
    import torch
    pytorch = True
    print('pytorch found')
except ModuleNotFoundError as err:
    #print(err)
    pytorch = False

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.interfaces.cache.enable()
    fftw = True
    print(f'fftw found using {pyfftw.config.NUM_THREADS} threads')
except:
    import numpy.fft as fft

try:
    import cv2

    opencv = True
except:
    opencv = False

try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.ndimage
    from cupyx.scipy.ndimage.filters import gaussian_filter as gaussian_filter_cupy

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    cp.fft.config.get_plan_cache().set_size(0)
    cupy = True
except:
    cupy = False

class BaseSimProcessor3D(BaseSimProcessor):
    def __init__(self):
        super().__init__()

    def calibrate(self, img, findCarrier=True):
        self._calibrate(img, findCarrier)

    def calibrate_cupy(self, img, findCarrier=True):
        assert cupy, "No CuPy present"
        self._calibrate(img, findCarrier, useCupy=True)

    def calibrate_pytorch(self, img, findCarrier=True):
        assert pytorch, "No pytorch present"
        self._calibrate(img, findCarrier, useTorch=True)

    def _calibrate(self, img, findCarrier=True, useTorch=False, useCupy=False):
        assert len(img) > self._nsteps - 1
        self.empty_cache()
        self.Ny, self.Nx = img[0, :, :].shape
        if self.Nx != self._lastN[0] and self.Ny != self._lastN[1]:
            self._allocate_arrays()

        ''' define k grids '''
        self._dx = self.pixelsize / self.magnification  # Sampling in image plane
        self._dy = self._dx
        self._res = self.wavelength / (2 * self.NA)
        self._oversampling = self._res / self._dx
        self._dkx = self._oversampling / (self.Nx / 2)  # Sampling in frequency plane
        self._dky = self._oversampling / (self.Ny / 2)  # Sampling in frequency plane
        self._kx = np.arange(-self.Nx / 2, self.Nx / 2, dtype=np.double) * self._dkx
        self._ky = np.arange(-self.Ny / 2, self.Ny / 2, dtype=np.double) * self._dky
        self._dx2 = self._dx / 2
        self._dy2 = self._dx2

        self._kr = np.sqrt(self._kx ** 2 + self._ky[:,np.newaxis] ** 2, dtype=np.single)
        kxbig = np.arange(-self.Nx, self.Nx, dtype=np.single) * self._dkx
        kybig = np.arange(-self.Ny, self.Ny, dtype=np.single) * self._dky
        kybig = kybig[:,np.newaxis]

        '''Sum input images if there are more than self._nsteps'''
        Nz = len(img) // self._nsteps
        if len(img) > self._nsteps:
            imgsum = np.zeros((self._nsteps, self.Ny, self.Nx), dtype=np.single)
            for i in range(self._nsteps):
                imgsum[i, :, :] = np.sum(img[i:(len(img) // self._nsteps) * self._nsteps:self._nsteps, :, :], 0, dtype = np.single)
        else:
            imgsum = np.single(img)
        imgs = np.single(img).reshape(Nz, self._nsteps, self.Nx, self.Ny)

        '''Separate bands into DC and high frequency bands'''
        self._M = np.linalg.pinv(self._get_band_construction_matrix())

        wienerfilter = np.zeros((2 * self.Ny, 2 * self.Nx), dtype=np.single)

        if useTorch:
            sum_prepared_comp = torch.einsum('ij,jkl->ikl', torch.as_tensor(self._M[:self._nbands + 1, :], device=self.tdev),
                                          torch.as_tensor(imgsum, dtype=torch.complex64, device=self.tdev)).cpu().numpy()
        elif useCupy:
            sum_prepared_comp = cp.dot(cp.asarray(self._M[:self._nbands + 1, :]),
                                           cp.asarray(imgsum).transpose((1, 0, 2))).get()
        else:
            prepared_comp = np.einsum('ik, jklm -> ijlm', self._M[:self._nbands + 1, :], imgs)

        # find parameters
        ckx = np.zeros((self._nbands, 1), dtype=np.single)
        cky = np.zeros((self._nbands, 1), dtype=np.single)
        p = np.zeros((self._nbands, 1), dtype=np.single)
        ampl = np.zeros((self._nbands, 1), dtype=np.single)

        if findCarrier:
            # minimum search radius in k-space
            for i in range(self._nbands):
                mask1 = (self._kr > 1.9 * self.eta * self.etafac[i])
                if useTorch:
                    self.kx[i], self.ky[i] = self._coarseFindCarrier_pytorch(sum_prepared_comp[0, :, :],
                                                            sum_prepared_comp[i + 1, :, :], mask1, self.eta * self.etafac[i])
                elif useCupy:
                    self.kx[i], self.ky[i] = self._coarseFindCarrier_cupy(sum_prepared_comp[0, :, :],
                                                            sum_prepared_comp[i + 1, :, :], mask1, self.eta * self.etafac[i])
                else:
                    self.kx[i], self.ky[i] = self._coarseFindCarrier(prepared_comp[0],
                                                            prepared_comp[i + 1], mask1, self.eta * self.etafac[i])
        for i in range(self._nbands):
            if useTorch:
                ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_pytorch(sum_prepared_comp[0, :, :],
                                                                    sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                    self.ky[i], self.eta * self.etafac[i])
            elif useCupy:
                ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_cupy(sum_prepared_comp[0, :, :],
                                                                    sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                    self.ky[i], self.eta * self.etafac[i])
            else:
                ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier(prepared_comp[0],
                                                                    prepared_comp[i + 1], self.kx[i],
                                                                    self.ky[i], self.eta * self.etafac[i])
            # For now compensate for 3D carrier component amplitude under-estimate with a simple 4x scaling factor.
            if self.etafac[i] < 1.0:
                ampl[i] *= 4.0

        self.kx = ckx # store found kx, ky, p and ampl values
        self.ky = cky
        self.p = p
        self.ampl = ampl

        if self.debug:
            print(f'kx = {tuple(ckx.flat)}')
            print(f'ky = {tuple(cky.flat)}')
            print(f'p  = {tuple(p.flat)}')
            print(f'a  = {tuple(ampl.flat)}')

        # Measure and correct for carrier phases

        if self.usePhases:
            self.phi = np.zeros((self._nbands, self._nsteps), dtype=np.single)
            for i in range(self._nbands):
                self.phi[i, :], _ = self.find_phase(self.kx[i], self.ky[i], img, useCupy=useCupy, useTorch=useTorch)
            if self.debug:
                print(self.phi)
                print(self.phi.shape)
            self._M = np.linalg.pinv(self._get_band_construction_matrix(self.phi))

            # sum_prepared_comp.fill(0)
            if useTorch:
                sum_prepared_comp = torch.einsum('ij,jkl->ikl',
                                                 torch.as_tensor(self._M[:self._nbands + 1, :], device=self.tdev),
                                                 torch.as_tensor(imgsum, dtype=torch.complex64, device=self.tdev) + 0 * 1j).cpu().numpy()
            elif useCupy:
                sum_prepared_comp = cp.dot(cp.asarray(self._M[:self._nbands+1, :]), cp.asarray(imgsum).transpose((1, 0, 2))).get()
            else:
                prepared_comp = np.einsum('ik, jklm -> ijlm', self._M[:self._nbands + 1, :], imgs)

            for i in range(self._nbands):
                if useTorch:
                    ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_pytorch(sum_prepared_comp[0, :, :],
                                                                                sum_prepared_comp[i + 1, :, :],
                                                                                self.kx[i], self.ky[i],
                                                                                self.eta * self.etafac[i])
                elif useCupy:
                    ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_cupy(sum_prepared_comp[0, :, :],
                                                                             sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                             self.ky[i], self.eta * self.etafac[i])
                else:
                    ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier(prepared_comp[0],
                                                                        prepared_comp[i + 1], self.kx[i],
                                                                        self.ky[i], self.eta * self.etafac[i])
                # For now compensate for 3D carrier component amplitude under-estimate with a simple 4x scaling factor.
                if self.etafac[i] < 1.0:
                    ampl[i] *= 4.0
            self.kx = ckx  # store found kx, ky, p and ampl values
            self.ky = cky
            self.p = p
            self.ampl = ampl
            if self.debug:
                print(f'kx = {ckx}')
                print(f'ky = {cky}')
                print(f'p  = {p}')
                print(f'a  = {ampl}')
        else:
            self.phi = None

        ph = np.single(2 * pi * self.NA / self.wavelength)

        xx = np.arange(-self.Nx, self.Nx, dtype=np.single) * self._dx2
        yy = np.arange(-self.Ny, self.Ny, dtype=np.single) * self._dy2

        if useTorch:
            Mcp = torch.as_tensor(self._M, device=self.tdev)
            Acp = torch.as_tensor(ampl, device=self.tdev)
            for i in range(self._nsteps):
                self._reconfactor[i, :, :] = (Mcp[0, i].real + torch.sum(torch.stack([torch.outer(
                    torch.exp(torch.as_tensor(1j * (ph * cky[j] * yy + p[j]), device=self.tdev)) * (Mcp[j + 1, i] * 4 / Acp[j]),
                    torch.exp(torch.as_tensor(1j * ph * ckx[j] * xx, device=self.tdev))).real for j in range(self._nbands)]), 0)).cpu().numpy()
        elif useCupy:
            Mcp = cp.asarray(self._M)
            Acp = cp.asarray(ampl)
            for i in range(self._nsteps):
                self._reconfactor[i, :, :] = (Mcp[0, i].real + cp.sum(cp.array([cp.outer(
                    cp.exp(cp.asarray(1j * (ph * cky[j] * yy + p[j]))) * (Mcp[j + 1, i] * 4 / Acp[j]),
                    cp.exp(cp.asarray(1j * ph * ckx[j] * xx))).real for j in range(self._nbands)]), 0)).get()
        else:
            for i in range(self._nsteps):
                self._reconfactor[i, :, :] = self._M[0, i].real + np.sum([np.outer(
                    np.exp(1j * (ph * cky[j] * yy + p[j])) * (self._M[j + 1, i] * 4 / ampl[j]),
                    np.exp(1j * ph * ckx[j] * xx)).real for j in range(self._nbands)], 0)

        if self.debug > 1:
            plt.figure()
            plt.title('Matrix')
            plt.imshow(np.abs(self._M))
            for i in range(self._nsteps):
                plt.figure()
                plt.title(f'reconfactor[{i}]')
                plt.imshow(self._reconfactor[i,500:540,500:540])

        # calculate pre-filter factors

        mask2 = (self._kr < 2)

        self._prefilter = np.single((self._tfm(self._kr, mask2) * self._attm(self._kr, mask2)))
        self._prefilter = fft.fftshift(self._prefilter)

        mtot = np.full((2 * self.Ny, 2 * self.Nx), False)

        thsteps = 361
        th = np.linspace(-pi, pi, thsteps, dtype=np.single)
        inv = np.geterr()['invalid']
        kmaxth = 2

        for i in range(0, self._nbands):
            krbig = sqrt((kxbig - ckx[i]) ** 2 + (kybig - cky[i]) ** 2)
            mask = (krbig < 2)
            mtot = mtot | mask
            wienerfilter[mask] = wienerfilter[mask] + (self._tf(krbig[mask]) ** 2) * self._att(krbig[mask])
            krbig = sqrt((kxbig + ckx[i]) ** 2 + (kybig + cky[i]) ** 2)
            mask = (krbig < 2)
            mtot = mtot | mask
            wienerfilter[mask] = wienerfilter[mask] + (self._tf(krbig[mask]) ** 2) * self._att(krbig[mask])
            np.seterr(invalid='ignore')  # Silence sqrt warnings for kmaxth calculations
            kmaxth = np.fmax(kmaxth, np.fmax(ckx[i] * np.cos(th) + cky[i] * np.sin(th) +
                                        np.sqrt(4 - (ckx[i] * np.sin(th)) ** 2 - (cky[i] * np.cos(th)) ** 2 +
                                                ckx[i] * cky[i] * np.sin(2 * th)),
                                        - ckx[i] * np.cos(th) - cky[i] * np.sin(th) +
                                        np.sqrt(4 - (ckx[i] * np.sin(th)) ** 2 - (cky[i] * np.cos(th)) ** 2 +
                                                ckx[i] * cky[i] * np.sin(2 * th))))
            np.seterr(invalid=inv)
        if self.debug:
            plt.figure()
            plt.plot(th, kmaxth)

        krbig = sqrt(kxbig ** 2 + kybig ** 2)
        mask = (krbig < 2)
        mtot = mtot | mask
        wienerfilter[mask] = (wienerfilter[mask] + self._tf(krbig[mask]) ** 2 * self._att(krbig[mask]))
        self.wienerfilter = wienerfilter

        if useTorch:  # interp not available in pytorch
            theta = torch.atan2(torch.as_tensor(kybig, device=self.tdev, dtype=torch.float32),
                                torch.as_tensor(kxbig, device=self.tdev, dtype=torch.float32))
            kmaxth_pt = torch.as_tensor(kmaxth, device=self.tdev)
            th_pt = torch.as_tensor(th, device=self.tdev)
            dth = th[1]-th[0]
            starti = torch.floor((theta + pi) / dth).long()
            start = kmaxth_pt[starti]
            end = kmaxth_pt[(starti+1) % (thsteps - 1)]
            weight = (theta - th_pt[starti]) / dth
            kmax = torch.lerp(start, end, weight).cpu().numpy()
        elif useCupy and 'interp' in dir(cp):  # interp not available in cupy version < 9.0.0
            kmax = cp.interp(cp.arctan2(cp.asarray(kybig), cp.asarray(kxbig)), cp.asarray(th), cp.asarray(kmaxth),
                                 period=2 * pi).astype(np.single).get()
        else:
            kmax = np.interp(np.arctan2(kybig, kxbig), th, kmaxth, period=2 * pi).astype(np.single)

        if self.debug:
            plt.figure()
            plt.title('WienerFilter')
            plt.imshow(wienerfilter)
            plt.figure()
            plt.title('output apodisation')
            plt.imshow(mtot * self._tf(1.99 * krbig * mtot / kmax, a_type = 'none'))
            plt.show()

        if useTorch:
            mtot_pt = torch.as_tensor(mtot, device=self.tdev)
            krbig_pt = torch.as_tensor(krbig, device=self.tdev)
            kmax_pt = torch.as_tensor(kmax, device=self.tdev)
            wienerfilter_pt = torch.as_tensor(wienerfilter, device=self.tdev)
            wienerfilter = (mtot_pt * self._tf_pytorch(1.99 * krbig_pt * mtot_pt / kmax_pt, a_type='none') /
                           (wienerfilter_pt * mtot_pt + self.w ** 2)).cpu().numpy()
        elif useCupy:
            mtot_cp = cp.asarray(mtot)
            wienerfilter = (mtot_cp * self._tf_cupy(1.99 * cp.asarray(krbig) * mtot_cp / cp.asarray(kmax),
                                                    a_type='none') /
                            (cp.asarray(wienerfilter) * mtot_cp + self.w ** 2)).get()
        else:
            wienerfilter = mtot * self._tf(1.99 * krbig * mtot / kmax, a_type = 'none') / \
                           (wienerfilter * mtot + self.w ** 2)
        self._postfilter = fft.fftshift(wienerfilter)

        if opencv:
            self._reconfactorU = [cv2.UMat(self._reconfactor[idx_p, :, :]) for idx_p in range(0, self._nsteps)]
            self._prefilter_ocv = np.single(cv2.dft(fft.ifft2(self._prefilter).real))
            pf = np.zeros((self.Ny, self.Nx, 2), dtype=np.single)
            pf[:, :, 0] = self._prefilter
            pf[:, :, 1] = self._prefilter
            self._prefilter_ocvU = cv2.UMat(np.single(pf))
            self._postfilter_ocv = np.single(cv2.dft(fft.ifft2(self._postfilter).real))
            pf = np.zeros((2 * self.Ny, 2 * self.Nx, 2), dtype=np.single)
            pf[:, :, 0] = self._postfilter
            pf[:, :, 1] = self._postfilter
            self._postfilter_ocvU = cv2.UMat(np.single(pf))

        if cupy:
            self._postfilter_cp = cp.asarray(self._postfilter)
        if pytorch:
            self._postfilter_torch = torch.as_tensor(self._postfilter, device=self.tdev)

    def crossCorrelations(self, img):
        '''
        Calculate cross-correlations to reveal carrier
        Sum cross correlations from input images if there are more than self._nsteps
        '''
        Ny, Nx = img[0, :, :].shape

        # Recalculate arrays by default to account for changes to parameters
        dx = self.pixelsize / self.magnification  # Sampling in image plane
        res = self.wavelength / (2 * self.NA)
        oversampling = res / dx
        dkx = oversampling / (Nx / 2)  # Sampling in frequency plane
        dky = oversampling / (Ny / 2)  # Sampling in frequency plane
        kx = np.arange(-Nx / 2, Nx / 2, dtype=np.double) * dkx
        ky = np.arange(-Ny / 2, Ny / 2, dtype=np.double) * dky
        kr = np.sqrt(kx ** 2 + ky[:, np.newaxis] ** 2, dtype=np.single)
        M = np.linalg.pinv(self._get_band_construction_matrix(self.phi))

        Nz = len(img) // self._nsteps
        imgs = img[:Nz * self._nsteps].astype(np.single).reshape((Nz, self._nsteps, self.Nx, self.Ny))
        prepared_comp = np.einsum('ik, jklm -> ijlm', self._M[:self._nbands + 1, :], imgs)

        ix = np.zeros((self._nbands, Nx, Ny), dtype=np.complex64)
        for i in range(self._nbands):
            otf_exclude_min_radius = self.eta * self.etafac[i] / 2
            # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            otf_exclude_max_radius = self.eta * self.etafac[i] * 2
            # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            maskbpf = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)

            motf = fft.fftshift(maskbpf / (self._tfm(kr, maskbpf) + (1 - maskbpf) * 0.0001))

            band0_common = fft.ifft2(fft.fft2(prepared_comp[0, :, :, :]) * motf)
            band1_common = fft.ifft2(fft.fft2(np.conjugate(prepared_comp[i + 1, :, :, :])) * motf)
            ix[i, :, :] = np.sum(band0_common * band1_common, axis=0)
        ixf = np.abs(fft.fftshift(fft.fft2(ix), axes=(1,2)))
        return ixf

    def crossCorrelations_cupy(self, img):
        '''Calculate cross-correlations to reveal carrier, sum input images if there are more than self._nsteps'''
        assert cupy, "No CuPy present"
        Ny, Nx = img[0, :, :].shape

        # Recalculate arrays by default to account for changes to parameters
        dx = self.pixelsize / self.magnification  # Sampling in image plane
        res = self.wavelength / (2 * self.NA)
        oversampling = res / dx
        dkx = oversampling / (Nx / 2)  # Sampling in frequency plane
        dky = oversampling / (Ny / 2)  # Sampling in frequency plane
        kx = cp.arange(-Nx / 2, Nx / 2, dtype=cp.double) * dkx
        ky = cp.arange(-Ny / 2, Ny / 2, dtype=cp.double) * dky
        kr = cp.sqrt(kx ** 2 + ky[:, cp.newaxis] ** 2, dtype=cp.single)
        M = cp.linalg.pinv(cp.asarray(self._get_band_construction_matrix(self.phi)))

        if len(img) > self._nsteps:
            imgs = cp.zeros((self._nsteps, Ny, Nx), dtype=np.single)
            imgt = cp.asarray(img, dtype=cp.single)
            for i in range(self._nsteps):
                imgs[i, :, :] = cp.sum(imgt[i:(len(img) // self._nsteps) * self._nsteps:self._nsteps, :, :], 0,
                                       dtype=cp.single)
        else:
            imgs = cp.asarray(img, dtype=cp.single)
        sum_prepared_comp = cp.dot(M[:self._nbands + 1, :], imgs.transpose((1, 0, 2)))
        ix = cp.zeros_like(sum_prepared_comp[1:, :, :], dtype=np.complex64)
        for i in range(self._nbands):
            otf_exclude_min_radius = self.eta * self.etafac[i] / 2  # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            otf_exclude_max_radius = self.eta * self.etafac[i] * 2  # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            maskbpf = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)

            motf = fft.fftshift(maskbpf / (self._tfm_cupy(kr, maskbpf) + (1 - maskbpf) * 0.0001))

            band0_common = cp.fft.ifft2(cp.fft.fft2(sum_prepared_comp[0, :, :]) * motf)
            band1_common = cp.fft.fft2(cp.fft.fft2(cp.conjugate(sum_prepared_comp[i + 1, :, :])) * motf)
            ix[i, :, :] = band0_common * band1_common

        ixf = cp.abs(cp.fft.fftshift(cp.fft.fft2(ix), axes=(1,2))).get()
        return ixf

    def crossCorrelations_pytorch(self, img):
        '''Calculate cross-correlations to revieal carrer, sum input images if there are more than self._nsteps'''
        assert pytorch, "No pytorch present"

        Ny, Nx = img[0, :, :].shape

        # Recalculate arrays by default to account for changes to parameters
        dx = self.pixelsize / self.magnification  # Sampling in image plane
        res = self.wavelength / (2 * self.NA)
        oversampling = res / dx
        dkx = oversampling / (Nx / 2)  # Sampling in frequency plane
        dky = oversampling / (Ny / 2)  # Sampling in frequency plane
        kx = torch.arange(-Nx / 2, Nx / 2, dtype=torch.float32, device=self.tdev) * dkx
        ky = torch.arange(-Ny / 2, Ny / 2, dtype=torch.float32, device=self.tdev) * dky
        kr = torch.sqrt(kx ** 2 + ky[:, np.newaxis] ** 2)
        M = torch.linalg.pinv(torch.as_tensor(self._get_band_construction_matrix(self.phi), dtype=torch.complex64, device=self.tdev))

        if len(img) > self._nsteps:
            imgs = torch.zeros((self._nsteps, Ny, Nx), dtype=torch.float32, device=self.tdev)
            imgt = torch.as_tensor(np.float32(img), device=self.tdev)
            for i in range(self._nsteps):
                imgs[i, :, :] = torch.sum(imgt[i:(len(img) // self._nsteps) * self._nsteps:self._nsteps, :, :], 0)
        else:
            imgs = torch.as_tensor(np.float32(img), device=self.tdev)
        sum_prepared_comp = torch.einsum('ij,jkl->ikl',M[:self._nbands + 1, :], imgs + 0 * 1j)
        ix = torch.zeros_like(sum_prepared_comp[1:, :, :], dtype=torch.complex64)
        for i in range(self._nbands):
            otf_exclude_min_radius = self.eta * self.etafac[i] / 2  # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            otf_exclude_max_radius = self.eta * self.etafac[i] * 2  # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            maskbpf = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)

            motf = torch.fft.fftshift(maskbpf / (self._tfm_pytorch(kr, maskbpf) + (~maskbpf) * 0.0001))

            band0_common = torch.fft.ifft2(torch.fft.fft2(sum_prepared_comp[0, :, :]) * motf)
            band1_common = torch.fft.ifft2(torch.fft.fft2(torch.conj(sum_prepared_comp[i + 1, :, :])) * motf)
            ix[i, :, :] = band0_common * band1_common

        ixf = torch.abs(torch.fft.fftshift(torch.fft.fft2(ix), dim=(1,2))).cpu().numpy()
        return ixf

    def find_phase_pytorch(self, kx, ky, img):
        return self.find_phase(kx, ky, img, useTorch=True)

    def find_phase_cupy(self, kx, ky, img):
        return self.find_phase(kx, ky, img, useCupy=True)

    def find_phase(self, kx, ky, img, useCupy=False, useTorch=False):
        # Finds the complex correlation coefficients of the spatial frequency component at (kx, ky)
        # in the images in img.  Run after calibration with a full set to find kx and estimated band0 component
        # Combines the given images modulo the self._nsteps, and returns self._nsteps amplitude and phase values
        assert (img.shape[0] >= self._nsteps), f'number of images in find_phase should be >= {self._nsteps}'
        nimages = (img.shape[0] // self._nsteps) * self._nsteps
        (kx, ky) = (kx.squeeze(), ky.squeeze())
        if useTorch:
            img = torch.as_tensor(np.single(img), device=self.tdev)
            kr = torch.as_tensor(self._kr, device=self.tdev)
            otf = torch.ones((self.Ny, self.Nx), dtype=torch.float32, device=self.tdev)
            imgnsum = torch.zeros((self._nsteps, img.shape[1], img.shape[2]), dtype=torch.float32, device=self.tdev)
            xx = torch.arange(-self.Nx / 2 * self._dx, self.Nx / 2 * self._dx, self._dx, dtype=torch.float64, device=self.tdev)
            yy = torch.arange(-self.Ny / 2 * self._dy, self.Ny / 2 * self._dy, self._dy, dtype=torch.float64, device=self.tdev)
            (TF, FFT, xp) = (self._tf_pytorch, torch.fft, torch)
        elif useCupy:
            img = cp.asarray(img, dtype=cp.float32)
            kr = cp.asarray(self._kr)
            otf = cp.ones((self.Ny, self.Nx), dtype=cp.float32)
            imgnsum = cp.zeros((self._nsteps, img.shape[1], img.shape[2]), dtype=cp.float32)
            xx = cp.arange(-self.Nx / 2 * self._dx, self.Nx / 2 * self._dx, self._dx, dtype=cp.float64)
            yy = cp.arange(-self.Ny / 2 * self._dy, self.Ny / 2 * self._dy, self._dy, dtype=cp.float64)
            (TF, FFT, xp) = (self._tf_cupy, cp.fft, cp)
        else:
            img = np.float32(img)
            kr = self._kr
            otf = np.ones((self.Ny, self.Nx), dtype = np.float32)
            imgnsum = np.zeros((self._nsteps, img.shape[1], img.shape[2]), dtype = np.single)
            xx = np.arange(-self.Nx / 2 * self._dx, self.Nx / 2 * self._dx, self._dx, dtype=np.double)
            yy = np.arange(-self.Ny / 2 * self._dy, self.Ny / 2 * self._dy, self._dy, dtype=np.double)
            (TF, FFT, xp) = (self._tf, np.fft, np)

        k = np.sqrt(kx * kx + ky * ky)
        hpf = 1.0 * (kr > 0.2 * k)
        m = kr < 0.8 * k
        otf[m] = TF(kr[m])

        # hpf = 1.0 * (kr > self.eta / 2)
        # m = kr < self.eta * 2
        # otf[m] = TF(kr[m])

        hpf[m] = hpf[m] / otf[m]
        hpf[~m] = 0
        hpf = FFT.fftshift(hpf)

        imgsum = xp.sum(img[:nimages, :, :], 0) / nimages
        for i in range(self._nsteps):
            imgnsum[i, :, :] = xp.sum(img[i:nimages:self._nsteps, :, :], 0) * self._nsteps / nimages

        p = FFT.ifft2(FFT.fft2(imgnsum - imgsum) * hpf)
        p0 = FFT.ifft2(FFT.fft2(imgsum) * hpf)
        phase_shift_to_xpeak = xp.exp(1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = xp.exp(1j * ky * yy * 2 * pi * self.NA / self.wavelength)
        scaling = 1 / xp.sum(p0 * p0.conj())
        cross_corr_result = xp.sum(p * p0 * xp.outer(
                        phase_shift_to_ypeak, phase_shift_to_xpeak), axis = (1,2)) * scaling

        if self.debug > 1:
            plt.figure()
            plt.title('Find phase')
            if useTorch:
                plt.imshow(xp.sqrt(xp.abs(FFT.fftshift(FFT.fft2(p[0, :, :] * p0)))).cpu().numpy(), cmap=plt.get_cmap('gray'))
            elif useCupy:
                plt.imshow(xp.sqrt(xp.abs(FFT.fftshift(FFT.fft2(p[0, :, :] * p0)))).get, cmap=plt.get_cmap('gray'))
            else:
                plt.imshow(xp.sqrt(xp.abs(FFT.fftshift(FFT.fft2(p[0, :, :] * p0)))), cmap=plt.get_cmap('gray'))
            ax = plt.gca()
            pxc0 = np.int32(np.round(kx / self._dkx) + self.Nx / 2)
            pyc0 = np.int32(np.round(ky / self._dky) + self.Ny / 2)
            circle = plt.Circle((pxc0, pyc0), color='red', fill=False)
            ax.add_artist(circle)
            mag = (25 * self.Ny / 256, 25 * self.Nx / 256)
            if useTorch:
                ixfz, Kx, Ky = self._zoomf(p0.cpu().numpy() * p[0, :, :].cpu().numpy(), (self.Ny, self.Nx), np.single(kx), np.single(ky), mag,
                                       self._dkx * self.Nx)
            elif useCupy:
                ixfz, Kx, Ky = self._zoomf(p0.get() * p[0, :, :].get(), (self.Ny, self.Nx), np.single(kx), np.single(ky), mag,
                                       self._dkx * self.Nx)
            else:
                ixfz, Kx, Ky = self._zoomf(p0 * p[0, :, :], (self.Ny, self.Nx), np.single(kx), np.single(ky), mag,
                                       self._dkx * self.Nx)
            plt.figure()
            plt.title('Zoom Find phase')
            plt.imshow(abs(ixfz.squeeze()))

        ampl = xp.abs(cross_corr_result) * 2
        phase = xp.angle(cross_corr_result)

        if self.debug > 1:
            if useTorch:
                print(f'found phase {phase.cpu().numpy()}, ampl {ampl.cpu().numpy()}')
            elif useCupy:
                print(f'found phase {phase.get()}, ampl {ampl.get()}')
            else:
                print(f'found phase {phase}, ampl {ampl}')

        if useTorch:
            return phase.cpu().numpy(), ampl.cpu().numpy()
        elif useCupy:
            return phase.get(), ampl.get()
        else:
            return phase, ampl

    def _coarseFindCarrier(self, band0, band1, mask, eta):
        otf_exclude_min_radius = eta / 2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        otf_exclude_max_radius = eta * 2 # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        maskbpf = (self._kr > otf_exclude_min_radius) & (self._kr < otf_exclude_max_radius)

        motf = fft.fftshift(maskbpf / (self._tfm(self._kr, maskbpf) + (1 - maskbpf) * 0.0001))

        band0_common = fft.ifft2(fft.fft2(band0) * motf)
        band1_common = fft.ifft2(fft.fft2(np.conjugate(band1)) * motf)
        # ix = band0_common * np.conjugate(band1_common)

        if band0_common.ndim == 2:
            ix = band0_common * band1_common
        else:
            ix = np.sum(band0_common * band1_common, axis=0)

        ixf = np.abs(fft.fftshift(fft.fft2(fft.fftshift(ix))))

        # pyc0, pxc0 = self._findPeak((ixf - gaussian_filter(ixf, 20)) * mask)
        pyc0, pxc0 = self._findPeak(ixf * mask)
        kx = self._dkx * (pxc0 - self.Nx / 2)
        ky = self._dky * (pyc0 - self.Ny / 2)

        if self.debug:
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf, cmap = plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = patches.Ellipse((self.Nx // 2, self.Ny // 2), width=4 / self._dkx, height=4 / self._dky, color='green', fill=False)
            ax.add_artist(circle)
            circle = patches.Ellipse((self.Nx // 2, self.Ny // 2), width=3.8 * eta / self._dkx, height=3.8 * eta / self._dky, color='cyan', fill=False)
            ax.add_artist(circle)

        return kx, ky

    def _refineCarrier(self, band0, band1, kx_in, ky_in, eta):
        pxc0 = np.int32(np.round(kx_in / self._dkx) + self.Nx // 2)
        pyc0 = np.int32(np.round(ky_in / self._dky) + self.Ny // 2)

        otf_exclude_min_radius = eta / 2
        otf_exclude_max_radius = eta * 2

        m = (self._kr < 2)
        otf = fft.fftshift(self._tfm(self._kr, m) + (1 - m) * 0.0001)

        otf_mask = (self._kr > otf_exclude_min_radius) & (self._kr < otf_exclude_max_radius)
        otf_mask_for_band_common_freq = fft.fftshift(
            otf_mask & scipy.ndimage.shift(otf_mask, (pyc0 - (self.Ny // 2 ), pxc0 - (self.Nx // 2 )), order=0))

        if self.debug:
            plt.figure()
            plt.title('Common mask')
            plt.imshow(fft.fftshift(otf_mask_for_band_common_freq), cmap=plt.get_cmap('gray'))

        band0_common = fft.ifft2(fft.fft2(band0) / otf * otf_mask_for_band_common_freq)

        band1_common = fft.ifft2(fft.fft2(np.conjugate(band1)) / otf * otf_mask_for_band_common_freq)

        if band0_common.ndim == 2:
            band = band0_common * band1_common
        else:
            band = np.sum(band0_common * band1_common, axis=0)

        if self.debug:
            ixf = np.abs(fft.fftshift(fft.fft2(fft.fftshift(band))))
            plt.figure()
            plt.title('Refine carrier')
            plt.imshow(ixf, cmap = plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = patches.Ellipse((self.Nx // 2, self.Ny // 2), width=4 / self._dkx, height=4 / self._dky, color='green', fill=False)
            ax.add_artist(circle)

        mag = (25 * self.Ny / 256, 25 * self.Nx / 256)
        ixfz, Kx, Ky = self._zoomf(band, (self.Nx, self.Ny), np.single(self._kx[pxc0]), np.single(self._ky[pyc0]), mag , self._dkx * self.Nx)
        pyc, pxc = self._findPeak(abs(ixfz))

        if self.debug:
            plt.figure()
            plt.title('Zoom Find carrier (ampl)')
            plt.imshow(np.abs(ixfz))
            plt.figure()
            plt.title('Zoom Find carrier (phase)')
            plt.imshow(np.angle(ixfz))

        kx = Kx[pxc]
        ky = Ky[pyc]

        xx = np.arange(-self.Nx / 2, self.Nx / 2, dtype=np.double) * self._dx
        yy = np.arange(-self.Ny / 2, self.Ny / 2, dtype=np.double) * self._dy
        phase_shift_to_xpeak = exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = exp(-1j * ky * yy * 2 * pi * self.NA / self.wavelength)

        scaling = 1 / np.sum(band0_common * np.conjugate(band0_common))
        cross_corr_result = np.sum(band0_common * band1_common * np.outer(
                        phase_shift_to_ypeak, phase_shift_to_xpeak)) * scaling

        ampl = np.abs(cross_corr_result) * 2
        phase = np.angle(cross_corr_result)
        return kx, ky, phase, ampl

    def _coarseFindCarrier_cupy(self, band0, band1, mask, eta):
        band0 = cp.asarray(band0)
        band1 = cp.asarray(band1)
        mask = cp.asarray(mask)
        kr = cp.asarray(self._kr)

        otf_exclude_min_radius = eta / 2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        otf_exclude_max_radius = eta * 2 # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        maskbpf = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)

        motf = cp.fft.fftshift(maskbpf / (self._tfm_cupy(kr, maskbpf) + (1 - maskbpf) * 0.0001))

        band0_common = cp.fft.ifft2(cp.fft.fft2(band0) * motf)
        band1_common = cp.fft.ifft2(cp.fft.fft2(cp.conjugate(band1)) * motf)
        ix = band0_common * band1_common

        ixf = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ix))))

        if self.debug:
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf.get(), cmap=plt.get_cmap('gray'))

        # pyc0, pxc0 = self._findPeak_cupy((ixf - gaussian_filter_cupy(ixf, 20)) * mask)
        pyc0, pxc0 = self._findPeak_cupy(ixf * mask)
        kx = self._dkx * (pxc0 - self.Nx / 2)
        ky = self._dky * (pyc0 - self.Ny / 2)

        return kx.get(), ky.get()

    def _refineCarrier_cupy(self, band0, band1, kx_in, ky_in, eta):
        band0 = cp.asarray(band0)
        band1 = cp.asarray(band1)

        pxc0 = np.int32(np.round(kx_in / self._dkx) + self.Nx // 2)
        pyc0 = np.int32(np.round(ky_in / self._dky) + self.Ny // 2)

        otf_exclude_min_radius = eta / 2
        otf_exclude_max_radius = eta * 2

        # kr = cp.sqrt(cp.asarray(self._kx) ** 2 + cp.asarray(self._ky) ** 2)
        kr = cp.asarray(self._kr, dtype=np.double)
        m = (kr < 2)
        otf = cp.fft.fftshift(self._tfm_cupy(kr, m) + (1 - m)*0.0001)

        otf_mask = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)
        otf_mask_for_band_common_freq = cp.fft.fftshift(
            otf_mask & cupyx.scipy.ndimage.shift(otf_mask, (pyc0 - (self.Ny // 2), pxc0 - (self.Nx // 2)), order=0))

        band0_common = cp.fft.ifft2(cp.fft.fft2(band0) / otf * otf_mask_for_band_common_freq)
        band1_common = cp.fft.ifft2(cp.fft.fft2(cp.conjugate(band1)) / otf * otf_mask_for_band_common_freq)

        band = band0_common * band1_common

        mag = (25 * self.Ny / 256, 25 * self.Nx / 256)
        ixfz, Kx, Ky = self._zoomf_cupy(band, (self.Ny, self.Nx), np.single(self._kx[pxc0]), np.single(self._ky[pyc0]), mag, self._dkx * self.Nx)
        pyc, pxc = self._findPeak_cupy(abs(ixfz))

        if self.debug:
            plt.figure()
            plt.title('Zoom Find carrier')
            plt.imshow(abs(ixfz.get()))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = patches.Ellipse((self.Nx // 2, self.Ny // 2), width=4 / self._dkx, height=4 / self._dky, color='green', fill=False)
            ax.add_artist(circle)

        kx = Kx[pxc]
        ky = Ky[pyc]

        xx = cp.arange(-self.Nx / 2, self.Nx / 2, dtype=np.double) * self._dx
        yy = cp.arange(-self.Ny / 2, self.Ny / 2, dtype=np.double) * self._dy
        phase_shift_to_xpeak = cp.exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = cp.exp(-1j * ky * yy * 2 * pi * self.NA / self.wavelength)

        scaling = 1 / cp.sum(band0_common * cp.conjugate(band0_common))

        cross_corr_result = cp.sum(band0_common * band1_common * cp.outer(
            phase_shift_to_ypeak, phase_shift_to_xpeak)) * scaling

        ampl = cp.abs(cross_corr_result) * 2
        phase = cp.angle(cross_corr_result)
        return kx.get(), ky.get(), phase.get(), ampl.get()

    def _coarseFindCarrier_pytorch(self, band0, band1, mask, eta):
        band0 = torch.as_tensor(band0, device=self.tdev)
        band1 = torch.as_tensor(band1, device=self.tdev)
        mask = torch.as_tensor(mask, device=self.tdev)
        kr = torch.as_tensor(self._kr, device=self.tdev)

        otf_exclude_min_radius = eta / 2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        otf_exclude_max_radius = eta * 2 # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        maskbpf = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)

        motf = torch.fft.fftshift(maskbpf / (self._tfm_pytorch(kr, maskbpf) + (~maskbpf) * 0.0001))

        band0_common = torch.fft.ifft2(torch.fft.fft2(band0) * motf)
        band1_common = torch.fft.ifft2(torch.fft.fft2(torch.conj(band1)) * motf)
        ix = band0_common * band1_common

        ixf = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(ix))))

        pyc0, pxc0 = self._findPeak_pytorch(ixf * mask)
        kx = (self._dkx * (pxc0 - self.Nx / 2)).cpu().numpy()
        ky = (self._dky * (pyc0 - self.Ny / 2)).cpu().numpy()

        if self.debug:
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf.cpu().numpy(), cmap=plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = patches.Ellipse((self.Nx // 2, self.Ny // 2), width=4 / self._dkx, height=4 / self._dky, color='green', fill=False)
            ax.add_artist(circle)

        return kx, ky

    def _refineCarrier_pytorch(self, band0, band1, kx_in, ky_in, eta):
        band0 = torch.as_tensor(band0, device=self.tdev)
        band1 = torch.as_tensor(band1, device=self.tdev)

        pxc0 = np.int32(np.round(kx_in / self._dkx) + self.Nx // 2)
        pyc0 = np.int32(np.round(ky_in / self._dky) + self.Ny // 2)

        otf_exclude_min_radius = eta / 2
        otf_exclude_max_radius = eta * 2

        # kr = cp.sqrt(cp.asarray(self._kx) ** 2 + cp.asarray(self._ky) ** 2)
        kr = torch.as_tensor(self._kr, dtype=torch.float, device=self.tdev)
        m = (kr < 2)
        otf = torch.fft.fftshift(self._tfm_pytorch(kr, m) + (~m)*0.0001)

        otf_mask = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)
        shiftx = int(pxc0 - (self.Nx // 2))
        shifty = int(pyc0 - (self.Ny // 2))
        otf_mask_shifted = torch.roll(otf_mask, shifts=(shifty, shiftx), dims=(0, 1))
        if shiftx < 0:
            otf_mask_shifted[:, shiftx:] = False
        else:
            otf_mask_shifted[:, :shiftx] = False
        if shifty < 0:
            otf_mask_shifted[shifty:, :] = False
        else:
            otf_mask_shifted[:shifty, :] = False
        otf_mask_for_band_common_freq = torch.fft.fftshift(otf_mask & otf_mask_shifted)

        if self.debug:
            plt.figure()
            plt.title('Common mask')
            plt.imshow(fft.fftshift(otf_mask_for_band_common_freq.cpu().numpy()), cmap=plt.get_cmap('gray'))

        band0_common = torch.fft.ifft2(torch.fft.fft2(band0) / otf * otf_mask_for_band_common_freq)
        band1_common = torch.fft.ifft2(torch.fft.fft2(torch.conj(band1)) / otf * otf_mask_for_band_common_freq)

        band = band0_common * band1_common

        if self.debug:
            ixf = np.abs(fft.fftshift(fft.fft2(fft.fftshift(band.cpu().numpy()))))
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf, cmap = plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = patches.Ellipse((self.Nx // 2, self.Ny // 2), width=4 / self._dkx, height=4 / self._dky, color='green', fill=False)
            ax.add_artist(circle)

        mag = (25 * self.Ny / 256, 25 * self.Nx / 256)
        ixfz, Kx, Ky = self._zoomf_pytorch(band, (self.Ny, self.Nx), np.single(self._kx[pxc0]).item(), np.single(self._ky[pyc0]).item(), mag, self._dkx * self.Nx)
        pyc, pxc = self._findPeak_pytorch(abs(ixfz))

        if self.debug:
            plt.figure()
            plt.title('Zoon Find carrier')
            plt.imshow(abs(ixfz.cpu().numpy()))

        kx = Kx[pxc]
        ky = Ky[pyc]

        xx = torch.arange(-self.Nx / 2, self.Nx / 2, dtype=torch.float, device=self.tdev) * self._dx
        yy = torch.arange(-self.Ny / 2, self.Ny / 2, dtype=torch.float, device=self.tdev) * self._dy
        phase_shift_to_xpeak = torch.exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = torch.exp(-1j * ky * yy * 2 * pi * self.NA / self.wavelength)

        scaling = 1 / torch.sum(band0_common * torch.conj(band0_common))

        cross_corr_result = torch.sum(band0_common * band1_common * torch.outer(
            phase_shift_to_ypeak, phase_shift_to_xpeak)) * scaling

        ampl = torch.abs(cross_corr_result) * 2
        phase = torch.angle(cross_corr_result)
        return kx.cpu().numpy(), ky.cpu().numpy(), phase.cpu().numpy(), ampl.cpu().numpy()