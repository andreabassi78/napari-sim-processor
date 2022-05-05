import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
from numpy import exp, pi, sqrt, log2, arccos
from scipy.ndimage import gaussian_filter

try:
    import torch
    pytorch = True
except ModuleNotFoundError as err:
    print(err)
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

class BaseSimProcessor:
    N = 256  # points to use in fft
    pixelsize = 6.5  # camera pixel size, um
    magnification = 60  # objective magnification
    NA = 1.1        # numerial aperture at sample
    n = 1.33        # refractive index at sample
    wavelength = 0.488  # wavelength, um
    alpha = 0.3     # zero order attenuation width
    beta = 0.95     # zero order attenuation
    w = 0.3         # Wiener parameter
    eta = 0.75      # eta is the factor by which the illumination grid frequency
    a = 0.25         # otf attenuation factor (a = 1 gives no correction)
    a_type = 'none'   # otf attenuation type ( 'exp' or 'sph' or 'none')
    # exceeds the incoherent cutoff, eta=1 for normal SIM, eta=sqrt(3)/2 to maximise
    # resolution without zeros in TF carrier is 2*kmax*eta
    debug = True    # Set to False (or 0) for no debug information,
                    # to True (or 1) for minimal information,
                    # to integers > 1 for extra information
    tdev = None     # torch device to use - if none, then will choose gpu if available and cpu otherwise.
    usePhases = False   # Whether to measure and use individual phases in calibration/reconstruction
    _lastN = 0      # To track changes in array size that will force array re-allocation

    def __init__(self):
        # self._nsteps = 0
        # self._nbands = 0
        self.kx = np.zeros((self._nbands, 1), dtype=np.single)
        self.ky = np.zeros((self._nbands, 1), dtype=np.single)
        self.p = np.zeros((self._nbands, 1), dtype=np.single)
        self.ampl = np.zeros((self._nbands, 1), dtype=np.single)

    def _allocate_arrays(self):
        """ define matrix """
        self._reconfactor = np.zeros((self._nsteps, 2 * self.N, 2 * self.N), dtype=np.single)  # for reconstruction
        self._prefilter = np.zeros((self.N, self.N),
                                   dtype=np.single)  # for prefilter stage, includes otf and zero order supression
        self._postfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        self._carray = np.zeros((self._nsteps, 2 * self.N, 2 * self.N), dtype=np.complex64)
        self._carray1 = np.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=np.complex64)

        self._imgstore = np.zeros((self._nsteps, 2 * self.N, 2 * self.N), dtype=np.single)
        self._bigimgstore = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        if cupy:
            self._carray_cp = cp.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=np.complex64)
            self._bigimgstore_cp = cp.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        if pytorch:
            if self.tdev is None:
                if torch.has_cuda:
                    self.tdev = torch.device('cuda')
                else:
                    self.tdev = torch.device('cpu')
            self._carray_torch = torch.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=torch.complex64, device=self.tdev)
            self._bigimgstore_torch = torch.zeros((2 * self.N, 2 * self.N), dtype=torch.float32, device=self.tdev)
        if opencv:
            self._prefilter_ocv = np.zeros((self.N, self.N),
                                           dtype=np.single)  # for prefilter stage, includes otf and zero order supression
            self._postfilter_ocv = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_ocv = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
            self._carray_ocvU = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC2)
            self._bigimgstoreU = cv2.UMat(self._bigimgstore)
            self._imgstoreU = [cv2.UMat((self.N, self.N), s=0.0, type=cv2.CV_32F) for i in range(7)]
        self._lastN = self.N

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
        self.N = len(img[0, :, :])
        if self.N != self._lastN:
            self._allocate_arrays()

        ''' define k grids '''
        self._dx = self.pixelsize / self.magnification  # Sampling in image plane
        self._res = self.wavelength / (2 * self.NA)
        self._oversampling = self._res / self._dx
        self._dk = self._oversampling / (self.N / 2)  # Sampling in frequency plane
        self._k = np.arange(-self.N / 2, self.N / 2, dtype=np.double) * self._dk
        self._dx2 = self._dx / 2

        self._kr = np.sqrt(self._k ** 2 + self._k[:,np.newaxis] ** 2, dtype=np.single)
        kxbig = np.arange(-self.N, self.N, dtype=np.single) * self._dk
        kybig = kxbig[:,np.newaxis]

        '''Sum input images if there are more than self._nsteps'''
        if len(img) > self._nsteps:
            imgs = np.zeros((self._nsteps, self.N, self.N), dtype=np.single)
            for i in range(self._nsteps):
                imgs[i, :, :] = np.sum(img[i:(len(img) // self._nsteps) * self._nsteps:self._nsteps, :, :], 0, dtype = np.single)
        else:
            imgs = np.single(img)

        '''Separate bands into DC and high frequency bands'''
        self._M = np.linalg.pinv(self._get_band_construction_matrix())

        wienerfilter = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)

        if useTorch:
            sum_prepared_comp = torch.einsum('ij,jkl->ikl', torch.as_tensor(self._M[:self._nbands + 1, :], device=self.tdev),
                                          torch.as_tensor(imgs, dtype=torch.complex64, device=self.tdev)).cpu().numpy()
        elif useCupy:
            sum_prepared_comp = cp.dot(cp.asarray(self._M[:self._nbands + 1, :]),
                                           cp.asarray(imgs).transpose((1, 0, 2))).get()
        else:
            sum_prepared_comp = np.dot(self._M[:self._nbands + 1, :], imgs.transpose((1, 0, 2)))

        # find parameters
        ckx = np.zeros((self._nbands, 1), dtype=np.single)
        cky = np.zeros((self._nbands, 1), dtype=np.single)
        p = np.zeros((self._nbands, 1), dtype=np.single)
        ampl = np.zeros((self._nbands, 1), dtype=np.single)

        if findCarrier:
            # minimum search radius in k-space
            mask1 = (self._kr > 1.9 * self.eta)
            for i in range(self._nbands):
                if useTorch:
                    self.kx[i], self.ky[i] = self._coarseFindCarrier_pytorch(sum_prepared_comp[0, :, :],
                                                            sum_prepared_comp[i + 1, :, :], mask1)
                elif useCupy:
                    self.kx[i], self.ky[i] = self._coarseFindCarrier_cupy(sum_prepared_comp[0, :, :],
                                                            sum_prepared_comp[i + 1, :, :], mask1)
                else:
                    self.kx[i], self.ky[i] = self._coarseFindCarrier(sum_prepared_comp[0, :, :],
                                                            sum_prepared_comp[i + 1, :, :], mask1)
        for i in range(self._nbands):
            if useTorch:
                ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_pytorch(sum_prepared_comp[0, :, :],
                                                                    sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                    self.ky[i])
            elif useCupy:
                ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_cupy(sum_prepared_comp[0, :, :],
                                                                    sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                    self.ky[i])
            else:
                ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier(sum_prepared_comp[0, :, :],
                                                                    sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                    self.ky[i])

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
            phi = np.zeros((self._nbands, self._nsteps), dtype=np.single)
            for i in range(self._nbands):
                phi[i, :], _ = self.find_phase(self.kx[i], self.ky[i], img)
            if self.debug:
                print(phi)
                print(phi.shape)
            self._M = np.linalg.pinv(self._get_band_construction_matrix(phi))

            sum_prepared_comp.fill(0)
            if useTorch:
                sum_prepared_comp = torch.einsum('ij,jkl->ikl',
                                                 torch.as_tensor(self._M[:self._nbands + 1, :], device=self.tdev),
                                                 torch.as_tensor(imgs, dtype=torch.complex64, device=self.tdev) + 0 * 1j).cpu().numpy()
            elif useCupy:
                sum_prepared_comp = cp.dot(cp.asarray(self._M[:self._nbands+1, :]), cp.asarray(imgs).transpose((1, 0, 2))).get()
            else:
                sum_prepared_comp = np.dot(self._M[:self._nbands+1, :], imgs.transpose((1, 0, 2)))

            for i in range(self._nbands):
                if useTorch:
                    ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_pytorch(sum_prepared_comp[0, :, :],
                                                                                sum_prepared_comp[i + 1, :, :],
                                                                                self.kx[i],
                                                                                self.ky[i])
                elif useCupy:
                    ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier_cupy(sum_prepared_comp[0, :, :],
                                                                             sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                             self.ky[i])
                else:
                    ckx[i], cky[i], p[i], ampl[i] = self._refineCarrier(sum_prepared_comp[0, :, :],
                                                                        sum_prepared_comp[i + 1, :, :], self.kx[i],
                                                                        self.ky[i])
            self.kx = ckx  # store found kx, ky, p and ampl values
            self.ky = cky
            self.p = p
            self.ampl = ampl
            if self.debug:
                print(f'kx = {ckx}')
                print(f'ky = {cky}')
                print(f'p  = {p}')
                print(f'a  = {ampl}')

        ph = np.single(2 * pi * self.NA / self.wavelength)

        xx = np.arange(-self.N, self.N, dtype=np.single) * self._dx2
        yy = xx

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

        mtot = np.full((2 * self.N, 2 * self.N), False)

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

        if useTorch:
            mtot_pt = torch.as_tensor(mtot, device=self.tdev)
            krbig_pt = torch.as_tensor(krbig, device=self.tdev)
            kmax_pt = torch.as_tensor(kmax, device=self.tdev)
            wienerfilter_pt = torch.as_tensor(wienerfilter, device=self.tdev)
            wienerfilter = (mtot_pt * self._tf_pytorch(1.99 * krbig_pt * mtot_pt / kmax_pt, a_type='none') / \
                           (wienerfilter_pt * mtot_pt + self.w ** 2)).cpu().numpy()
        elif useCupy:
            mtot_cp = cp.asarray(mtot)
            wienerfilter = (mtot_cp * self._tf_cupy(1.99 * cp.asarray(krbig) * mtot_cp / cp.asarray(kmax),
                                                        a_type='none') / \
                                (cp.asarray(wienerfilter) * mtot_cp + self.w ** 2)).get()
        else:
            wienerfilter = mtot * self._tf(1.99 * krbig * mtot / kmax, a_type = 'none') / \
                           (wienerfilter * mtot + self.w ** 2)
        self._postfilter = fft.fftshift(wienerfilter)

        if opencv:
            self._reconfactorU = [cv2.UMat(self._reconfactor[idx_p, :, :]) for idx_p in range(0, self._nsteps)]
            self._prefilter_ocv = np.single(cv2.dft(fft.ifft2(self._prefilter).real))
            pf = np.zeros((self.N, self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._prefilter
            pf[:, :, 1] = self._prefilter
            self._prefilter_ocvU = cv2.UMat(np.single(pf))
            self._postfilter_ocv = np.single(cv2.dft(fft.ifft2(self._postfilter).real))
            pf = np.zeros((2 * self.N, 2 * self.N, 2), dtype=np.single)
            pf[:, :, 0] = self._postfilter
            pf[:, :, 1] = self._postfilter
            self._postfilter_ocvU = cv2.UMat(np.single(pf))

        if cupy:
            self._postfilter_cp = cp.asarray(self._postfilter)
        if pytorch:
            self._postfilter_torch = torch.as_tensor(self._postfilter, device=self.tdev)

    def WFreconstruct(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        wf_comp = np.einsum('j,kjlm->klm', self._M[0, :].real, imgr).squeeze()
        return wf_comp

    def filteredWFreconstruct(self, img):
        imgr = np.single(img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2]))
        wf_comp = np.einsum('j,kjlm->klm', self._M[0, :].real, imgr).squeeze()
        filt = fft.fftshift(self._attm(self._kr, self._kr < 2))
        filtered_wf_comp = fft.ifft2(fft.fft2(wf_comp) * filt)
        return filtered_wf_comp.real

    def WFreconstruct_cupy(self, img):
        imgr = cp.asarray(img).reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        wf_comp = cp.einsum('j,kjlm->klm', cp.asarray(self._M[0, :].real), imgr).squeeze()
        return wf_comp.get()

    def filteredWFreconstruct_cupy(self, img):
        imgr = cp.asarray(img).reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        wf_comp = cp.einsum('j,kjlm->klm', cp.asarray(self._M[0, :].real), imgr).squeeze()
        filt = cp.fft.fftshift(cp.asarray(self._attm(self._kr, self._kr < 2)))
        filtered_wf_comp = cp.fft.ifft2(cp.fft.fft2(wf_comp) * filt)
        return filtered_wf_comp.real.get()

    def WFreconstruct_pytorch(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        wf_comp = torch.einsum('j,kjlm->klm',
                                         torch.as_tensor(self._M[0, :], device=self.tdev),
                                         torch.as_tensor(np.float32(imgr), dtype=torch.float32,
                                                         device=self.tdev) + 0 * 1j).squeeze()
        return wf_comp.cpu().real.numpy()

    def filteredWFreconstruct_pytorch(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        wf_comp = torch.einsum('j,kjlm->klm',
                                         torch.as_tensor(self._M[0, :], device=self.tdev),
                                         torch.as_tensor(np.float32(imgr), dtype=torch.float32,
                                                         device=self.tdev) + 0 * 1j).squeeze()
        filt = torch.fft.fftshift(torch.as_tensor(self._attm(self._kr, self._kr < 2), device=self.tdev))
        filtered_wf_comp = torch.fft.ifft2(torch.fft.fft2(wf_comp) * filt)
        return filtered_wf_comp.real.cpu().numpy()

    def OSreconstruct(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        sum_prepared_comp = np.einsum('ij,kjlm->iklm', self._M[1:self._nbands + 1, :], imgr)
        imgos = np.einsum('iklm,ij->klmj', np.abs(sum_prepared_comp), 1 / self.ampl).squeeze()
        return imgos

    def filteredOSreconstruct(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        sum_prepared_comp = np.einsum('ij,kjlm->iklm', self._M[1:self._nbands + 1, :], imgr)
        filt = fft.fftshift(self._attm(self._kr, self._kr < 2))
        filtered_comp = fft.ifft2(fft.fft2(sum_prepared_comp) * filt)
        imgos = np.einsum('iklm,ij->klmj', np.abs(filtered_comp), 1 / self.ampl).squeeze()
        return imgos

    def OSreconstruct_cupy(self, img):
        imgr = cp.asarray(img).reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        sum_prepared_comp = cp.einsum('ij,kjlm->iklm', cp.asarray(self._M[1:self._nbands + 1, :]), imgr)
        imgos = cp.einsum('iklm,ij->klmj', cp.abs(sum_prepared_comp), cp.asarray(1 / self.ampl)).squeeze()
        return imgos.get()

    def filteredOSreconstruct_cupy(self, img):
        imgr = cp.asarray(img).reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        sum_prepared_comp = cp.einsum('ij,kjlm->iklm', cp.asarray(self._M[1:self._nbands + 1, :]), imgr)
        filt = cp.fft.fftshift(cp.asarray(self._attm(self._kr, self._kr < 2)))
        filtered_comp = cp.fft.ifft2(cp.fft.fft2(sum_prepared_comp) * filt)
        imgos = cp.einsum('iklm,ij->klmj', cp.abs(filtered_comp), cp.asarray(1 / self.ampl)).squeeze()
        return imgos.get()

    def OSreconstruct_pytorch(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        sum_prepared_comp = torch.einsum('ij,kjlm->iklm',
                                         torch.as_tensor(self._M[1:self._nbands + 1, :], device=self.tdev),
                                         torch.as_tensor(np.float32(imgr), dtype=torch.float32,
                                                         device=self.tdev) + 0 * 1j)
        imgos = torch.einsum('iklm,ij->klmj', torch.abs(sum_prepared_comp), 1 / torch.as_tensor(self.ampl, device=self.tdev)).squeeze()
        return imgos.cpu().numpy()

    def filteredOSreconstruct_pytorch(self, img):
        imgr = img.reshape(img.shape[0] // self._nsteps, self._nsteps, img.shape[1], img.shape[2])
        sum_prepared_comp = torch.einsum('ij,kjlm->iklm',
                                         torch.as_tensor(self._M[1:self._nbands + 1, :], device=self.tdev),
                                         torch.as_tensor(np.float32(imgr), dtype=torch.float32,
                                                         device=self.tdev) + 0 * 1j)
        filt = torch.fft.fftshift(torch.as_tensor(self._attm(self._kr, self._kr < 2), device=self.tdev))
        filtered_comp = torch.fft.ifft2(torch.fft.fft2(sum_prepared_comp) * filt)
        imgos = torch.einsum('iklm,ij->klmj', torch.abs(filtered_comp), 1 / torch.as_tensor(self.ampl, device=self.tdev)).squeeze()
        return imgos.cpu().numpy()

    def reconstruct_fftw(self, img):
        imf = fft.fft2(img) * self._prefilter
        self._carray[:, 0:self.N // 2, 0:self.N // 2] = imf[:, 0:self.N // 2, 0:self.N // 2]
        self._carray[:, 0:self.N // 2, 3 * self.N // 2:2 * self.N] = imf[:, 0:self.N // 2, self.N // 2:self.N]
        self._carray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2] = imf[:, self.N // 2:self.N, 0:self.N // 2]
        self._carray[:, 3 * self.N // 2:2 * self.N, 3 * self.N // 2:2 * self.N] = imf[:, self.N // 2:self.N,
                                                                                  self.N // 2:self.N]
        img2 = np.sum(np.real(fft.ifft2(self._carray)).real * self._reconfactor, 0)
        self._imgstore = img.copy()
        self._bigimgstore = fft.ifft2(fft.fft2(img2) * self._postfilter).real
        return self._bigimgstore

    def reconstruct_rfftw(self, img):
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[:, 0:self.N // 2, 0:self.N // 2 + 1]
        self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[:, self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = np.sum(fft.irfft2(self._carray1) * self._reconfactor, 0)
        self._imgstore = img.copy()
        self._bigimgstore = fft.irfft2(fft.rfft2(img2) * self._postfilter[:, 0:self.N + 1])
        return self._bigimgstore

    def reconstruct_ocv(self, img):
        assert opencv, "No opencv present"
        img2 = np.zeros((2 * self.N, 2 * self.N), dtype=np.single)
        for i in range(self._nsteps):
            imf = cv2.mulSpectrums(cv2.dft(img[i, :, :]), self._prefilter_ocv, 0)
            self._carray_ocv[0:self.N // 2, 0:self.N] = imf[0:self.N // 2, 0:self.N]
            self._carray_ocv[3 * self.N // 2:2 * self.N, 0:self.N] = imf[self.N // 2:self.N, 0:self.N]
            img2 = cv2.add(img2, cv2.multiply(cv2.idft(self._carray_ocv, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                                              self._reconfactor[i, :, :]))
        self._imgstore = img.copy()
        return cv2.idft(cv2.mulSpectrums(cv2.dft(img2), self._postfilter_ocv, 0),
                        flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    def reconstruct_ocvU(self, img):
        assert opencv, "No opencv present"
        img2 = cv2.UMat((2 * self.N, 2 * self.N), s=0.0, type=cv2.CV_32FC1)
        mask = cv2.UMat((self.N // 2, self.N // 2), s=1, type=cv2.CV_8U)
        for i in range(self._nsteps):
            self._imgstoreU[i] = cv2.UMat(img[i, :, :])
            imf = cv2.multiply(cv2.dft(self._imgstoreU[i], flags=cv2.DFT_COMPLEX_OUTPUT), self._prefilter_ocvU)
            cv2.copyTo(src=cv2.UMat(imf, (0, 0, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (0, 0, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (0, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (0, 3 * self.N // 2, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, 0, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 0, self.N // 2, self.N // 2)))
            cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                       dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 3 * self.N // 2, self.N // 2, self.N // 2)))
            img2 = cv2.add(img2, cv2.multiply(cv2.idft(self._carray_ocvU, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                                              self._reconfactorU[i]))
        self._bigimgstoreU = cv2.idft(cv2.multiply(cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT), self._postfilter_ocvU),
                                      flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return self._bigimgstoreU

    def reconstruct_cupy(self, img):
        assert cupy, "No CuPy present"
        self._imgstore = img.copy()
        imf = cp.fft.rfft2(cp.asarray(img)) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])
        self._carray_cp[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[:, 0:self.N // 2, 0:self.N // 2 + 1]
        self._carray_cp[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[:, self.N // 2:self.N,
                                                                            0:self.N // 2 + 1]
        del imf
        cp._default_memory_pool.free_all_blocks()
        img2 = cp.sum(cp.fft.irfft2(self._carray_cp) * cp.asarray(self._reconfactor), 0)
        self._bigimgstore_cp = cp.fft.irfft2(cp.fft.rfft2(img2) * self._postfilter_cp[:, 0:self.N + 1])
        del img2
        cp._default_memory_pool.free_all_blocks()
        return self._bigimgstore_cp.get()

    def reconstruct_pytorch(self, img):
        assert torch, "No PyTorch present"
        img = np.float32(img)
        self._imgstore = img.copy()
        imf = torch.fft.rfft2(torch.as_tensor(img, device=self.tdev)) * torch.as_tensor(self._prefilter[:, 0:self.N // 2 + 1], device=self.tdev)
        self._carray_torch[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[:, 0:self.N // 2, 0:self.N // 2 + 1]
        self._carray_torch[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[:, self.N // 2:self.N,
                                                                            0:self.N // 2 + 1]
        img2 = torch.sum(torch.fft.irfft2(self._carray_torch) * torch.as_tensor(self._reconfactor, device=self.tdev), 0)
        self._bigimgstore_torch = torch.fft.irfft2(torch.fft.rfft2(img2) * self._postfilter_torch[:, 0:self.N + 1])
        return self._bigimgstore_torch.cpu().numpy()

    # region Stream reconstruction functions
    def reconstructframe_fftw(self, img, i):
        diff = img - self._imgstore[i, :, :]
        imf = fft.fft2(diff) * self._prefilter
        self._carray[0, 0:self.N // 2, 0:self.N // 2] = imf[0:self.N // 2, 0:self.N // 2]
        self._carray[0, 0:self.N // 2, 3 * self.N // 2:2 * self.N] = imf[0:self.N // 2, self.N // 2:self.N]
        self._carray[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2] = imf[self.N // 2:self.N, 0:self.N // 2]
        self._carray[0, 3 * self.N // 2:2 * self.N, 3 * self.N // 2:2 * self.N] = imf[self.N // 2:self.N,
                                                                                  self.N // 2:self.N]
        img2 = fft.ifft2(self._carray[0, :, :]).real * self._reconfactor[i, :, :]
        self._imgstore[i, :, :] = img.copy()
        self._bigimgstore = self._bigimgstore + fft.ifft2(fft.fft2(img2) * self._postfilter).real
        return self._bigimgstore

    def reconstructframe_rfftw(self, img, i):
        diff = img.astype(np.single) - self._imgstore[i, :, :].astype(np.single)
        imf = fft.rfft2(diff) * self._prefilter[:, 0:self.N // 2 + 1]
        self._carray1[0, 0:self.N // 2, 0:self.N // 2 + 1] = imf[0:self.N // 2, 0:self.N // 2 + 1]
        self._carray1[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = fft.irfft2(self._carray1[0, :, :]) * self._reconfactor[i, :, :]
        self._imgstore[i, :, :] = img.copy()
        self._bigimgstore = self._bigimgstore + fft.irfft2(fft.rfft2(img2) * self._postfilter[:, 0:self.N + 1])
        return self._bigimgstore

    def reconstructframe_ocv(self, img, i):
        assert opencv, "No opencv present"
        diff = img - self._imgstore[i, :, :]
        imf = cv2.mulSpectrums(cv2.dft(diff), self._prefilter_ocv, 0)
        self._carray_ocv[0:self.N // 2, 0:self.N] = imf[0:self.N // 2, 0:self.N]
        self._carray_ocv[3 * self.N // 2:2 * self.N, 0:self.N] = imf[self.N // 2:self.N, 0:self.N]
        img2 = cv2.multiply(cv2.idft(self._carray_ocv, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                            self._reconfactor[i, :, :])
        self._imgstore[i, :, :] = img.copy()
        self._bigimgstore = self._bigimgstore + cv2.idft(cv2.mulSpectrums(cv2.dft(img2), self._postfilter_ocv, 0),
                                                         flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return self._bigimgstore

    def reconstructframe_ocvU(self, img, i):
        assert opencv, "No opencv present"
        mask = cv2.UMat((self.N // 2, self.N // 2), s=1, type=cv2.CV_8U)
        imU = cv2.UMat(img)
        diff = cv2.subtract(imU, self._imgstoreU[i])
        imf = cv2.multiply(cv2.dft(diff, flags=cv2.DFT_COMPLEX_OUTPUT), self._prefilter_ocvU)
        cv2.copyTo(src=cv2.UMat(imf, (0, 0, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (0, 0, self.N // 2, self.N // 2)))
        cv2.copyTo(src=cv2.UMat(imf, (0, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (0, 3 * self.N // 2, self.N // 2, self.N // 2)))
        cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, 0, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 0, self.N // 2, self.N // 2)))
        cv2.copyTo(src=cv2.UMat(imf, (self.N // 2, self.N // 2, self.N // 2, self.N // 2)), mask=mask,
                   dst=cv2.UMat(self._carray_ocvU, (3 * self.N // 2, 3 * self.N // 2, self.N // 2, self.N // 2)))
        img2 = cv2.multiply(cv2.idft(self._carray_ocvU, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT),
                            self._reconfactorU[i])
        self._imgstoreU[i] = imU
        self._bigimgstoreU = cv2.add(self._bigimgstoreU,
                                     cv2.idft(cv2.multiply(cv2.dft(img2, flags=cv2.DFT_COMPLEX_OUTPUT),
                                                               self._postfilter_ocvU)
                                              , flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT))
        return self._bigimgstoreU

    def reconstructframe_cupy(self, img, i):
        assert cupy, "No CuPy present"
        diff = cp.asarray(img, dtype=np.single) - cp.asarray(self._imgstore[i, :, :], dtype=np.single)
        imf = cp.fft.rfft2(diff) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])
        self._carray_cp[0, 0:self.N // 2, 0:self.N // 2 + 1] = imf[0:self.N // 2, 0:self.N // 2 + 1]
        self._carray_cp[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = cp.fft.irfft2(self._carray_cp[0, :, :]) * cp.asarray(self._reconfactor[i, :, :])
        self._bigimgstore_cp = self._bigimgstore_cp + cp.fft.irfft2(
            cp.fft.rfft2(img2) * self._postfilter_cp[:, 0:self.N + 1])
        self._imgstore[i, :, :] = img.copy()
        del img2
        del imf
        cp._default_memory_pool.free_all_blocks()
        return self._bigimgstore_cp.get()

    def reconstructframe_pytorch(self, img, i):
        assert torch, "No CuPy present"
        diff = torch.as_tensor(img, dtype=torch.float32, device=self.tdev) - torch.as_tensor(self._imgstore[i, :, :], device=self.tdev)
        imf = torch.fft.rfft2(diff) * torch.as_tensor(self._prefilter[:, 0:self.N // 2 + 1], device=self.tdev)
        self._carray_torch[0, 0:self.N // 2, 0:self.N // 2 + 1] = imf[0:self.N // 2, 0:self.N // 2 + 1]
        self._carray_torch[0, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[self.N // 2:self.N, 0:self.N // 2 + 1]
        img2 = torch.fft.irfft2(self._carray_torch[0, :, :]) * torch.as_tensor(self._reconfactor[i, :, :], device=self.tdev)
        self._bigimgstore_torch = self._bigimgstore_torch + torch.fft.irfft2(
            torch.fft.rfft2(img2) * self._postfilter_torch[:, 0:self.N + 1])
        self._imgstore[i, :, :] = img.copy()
        return self._bigimgstore_torch.cpu().numpy()

    # endregion

    def batchreconstruct(self, img):
        nim = img.shape[0]
        r = np.mod(nim, 2 * self._nsteps)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, np.zeros((2 * self._nsteps - r, self.N, self.N), np.single)))
            nim = nim + 2 * self._nsteps - r
        nimg = nim // self._nsteps
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        img2 = np.zeros([nim, 2 * self.N, 2 * self.N], dtype=np.single)
        for i in range(0, nim, self._nsteps):
            self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, 0:self.N // 2, 0:self.N // 2 + 1]
            self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, self.N // 2:self.N,
                                                                              0:self.N // 2 + 1]
            img2[i:i + self._nsteps, :, :] = fft.irfft2(self._carray1) * self._reconfactor
        img3 = fft.irfft(fft.rfft(img2, nim, 0)[0:nimg // 2 + 1, :, :], nimg, 0)
        res = fft.irfft2(fft.rfft2(img3) * self._postfilter[:, :self.N + 1])
        return res

    def batchreconstructcompact(self, img, blocksize = 128):
        nim = img.shape[0]
        r = np.mod(nim, 2 * self._nsteps)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, np.zeros((2 * self._nsteps - r, self.N, self.N), np.single)))
            nim = nim + 2 * self._nsteps - r
        nimg = nim // self._nsteps
        imf = fft.rfft2(img) * self._prefilter[:, 0:self.N // 2 + 1]
        img2 = np.zeros([nim, 2 * self.N, 2 * self.N], dtype=np.single)
        for i in range(0, nim, self._nsteps):
            self._carray1[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, 0:self.N // 2, 0:self.N // 2 + 1]
            self._carray1[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, self.N // 2:self.N,
                                                                              0:self.N // 2 + 1]
            img2[i:i + self._nsteps, :, :] = fft.irfft2(self._carray1) * self._reconfactor
        img3 = np.zeros((nimg, 2 * self.N, 2 * self.N), dtype=np.single)

        for offs in range(0, 2 * self.N - blocksize, blocksize):
            imf = fft.rfft(img2[:, offs:offs + blocksize, 0:2 * self.N], nim, 0)[:nimg // 2 + 1, :, :]
            img3[:, offs:offs + blocksize, 0:2 * self.N] = fft.irfft(imf, nimg, 0)
        imf = fft.rfft(img2[:, offs + blocksize:2 * self.N, 0:2 * self.N], nim, 0)[:nimg // 2 + 1, :, :]
        img3[:, offs + blocksize:2 * self.N, 0:2 * self.N] = fft.irfft(imf, nimg, 0)

        res = fft.irfft2(fft.rfft2(img3) * self._postfilter[:, :self.N + 1])
        return res

    def batchreconstruct_cupy(self, img):
        assert cupy, "No CuPy present"
        cp._default_memory_pool.free_all_blocks()
        img = cp.asarray(img, dtype=np.float32)
        nim = img.shape[0]
        r = np.mod(nim, 2 * self._nsteps)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, cp.zeros((2 * self._nsteps - r, self.N, self.N), np.single)))
            nim = nim + 2 * self._nsteps - r
        nimg = nim // self._nsteps
        imf = cp.fft.rfft2(img) * cp.asarray(self._prefilter[:, 0:self.N // 2 + 1])

        del img
        # cp._default_memory_pool.free_all_blocks()
        #
        # if self.debug:
        #     print(mempool.used_bytes())
        #     print(mempool.total_bytes())

        img2 = cp.zeros((nim, 2 * self.N, 2 * self.N), dtype=np.single)
        bcarray = cp.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=np.complex64)
        reconfactor_cp = cp.asarray(self._reconfactor)
        for i in range(0, nim, self._nsteps):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + self._nsteps, :, :] = cp.fft.irfft2(bcarray) * reconfactor_cp

        del imf
        del bcarray
        del reconfactor_cp
        # cp._default_memory_pool.free_all_blocks()

        # if self.debug:
        #     print(mempool.used_bytes())
        #     print(mempool.total_bytes())

        img3 = cp.fft.irfft(cp.fft.rfft(img2, nim, 0)[0:nimg // 2 + 1, :, :], nimg, 0)
        del img2
        # cp._default_memory_pool.free_all_blocks()
        # if self.debug:
        #     print(mempool.used_bytes())
        #     print(mempool.total_bytes())
        res = (cp.fft.irfft2(cp.fft.rfft2(img3) * self._postfilter_cp[:, :self.N + 1])).get()
        del img3
        # cp._default_memory_pool.free_all_blocks()
        return res

    def batchreconstructcompact_cupy(self, img):
        assert cupy, "No CuPy present"
        cp._default_memory_pool.free_all_blocks()
        img = cp.array(img, dtype=np.float32)
        nim = img.shape[0]
        r = np.mod(nim, 2 * self._nsteps)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = cp.concatenate((img, cp.zeros((2 * self._nsteps - r, self.N, self.N), np.single)))
            nim = nim + 2 * self._nsteps - r
        nimg = nim // self._nsteps
        imf = cp.fft.rfft2(img) * cp.array(self._prefilter[:, 0:self.N // 2 + 1])

        del img
        # cp._default_memory_pool.free_all_blocks()

        img2 = cp.zeros((nim, 2 * self.N, 2 * self.N), dtype=np.single)
        bcarray = cp.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=np.complex64)
        reconfactor_cp = cp.array(self._reconfactor)
        for i in range(0, nim, self._nsteps):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + self._nsteps, :, :] = cp.fft.irfft2(bcarray) * reconfactor_cp

        del bcarray
        del reconfactor_cp
        # cp._default_memory_pool.free_all_blocks()

        img3 = cp.zeros((nimg, 2 * self.N, 2 * self.N), dtype=np.single)
        blocksize = 128
        for offs in range(0, 2*self.N - blocksize, blocksize):
            imf = cp.fft.rfft(img2[:, offs:offs + blocksize, 0:2 * self.N], nim, 0)[:nimg // 2 + 1, :, :]
            img3[:, offs:offs + blocksize, 0:2 * self.N] = cp.fft.irfft(imf, nimg, 0)
        imf = cp.fft.rfft(img2[:, offs + blocksize:2 * self.N, 0:2 * self.N], nim, 0)[:nimg // 2 + 1, :, :]
        img3[:, offs + blocksize:2 * self.N, 0:2 * self.N] = cp.fft.irfft(imf, nimg, 0)
        del img2
        del imf
        # cp._default_memory_pool.free_all_blocks()

        res = (cp.fft.irfft2(cp.fft.rfft2(img3) * self._postfilter_cp[:, :self.N + 1])).get()
        del img3
        # cp._default_memory_pool.free_all_blocks()

        return res

    def batchreconstructcompact_pytorch(self, img, blocksize = 128):
        assert pytorch, "No pytorch present"
        if torch.has_cuda:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        nim = img.shape[0]
        r = np.mod(nim, 2 * self._nsteps)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, np.zeros((2 * self._nsteps - r, self.N, self.N), img.dtype)))
            nim = nim + 2 * self._nsteps - r
        nimg = nim // self._nsteps
        img1 = torch.as_tensor(np.single(img), dtype=torch.float32, device=self.tdev)
        imf = torch.fft.rfft2(img1) * torch.as_tensor(self._prefilter[:, 0:self.N // 2 + 1], device=self.tdev)
        del img1
        img2 = torch.zeros((nim, 2 * self.N, 2 * self.N), dtype=torch.float32, device=self.tdev)
        bcarray = torch.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=torch.complex64, device=self.tdev)
        reconfactor_pt = torch.as_tensor(self._reconfactor, device=self.tdev)
        for i in range(0, nim, self._nsteps):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + self._nsteps, :, :] = torch.fft.irfft2(bcarray) * reconfactor_pt

        img3 = torch.zeros((nimg, 2 * self.N, 2 * self.N), dtype=torch.float32, device=self.tdev)
        for offs in range(0, 2 * self.N - blocksize, blocksize):
            imf = torch.fft.rfft(img2[:, offs:offs + blocksize, 0:2 * self.N], nim, 0)[:nimg // 2 + 1, :, :]
            img3[:, offs:offs + blocksize, 0:2 * self.N] = torch.fft.irfft(imf, nimg, 0)
        imf = torch.fft.rfft(img2[:, offs + blocksize:2 * self.N, 0:2 * self.N], nim, 0)[:nimg // 2 + 1, :, :]
        img3[:, offs + blocksize:2 * self.N, 0:2 * self.N] = torch.fft.irfft(imf, nimg, 0)
        del img2
        postfilter_pt = torch.as_tensor(self._postfilter, device=self.tdev)
        res = (torch.fft.irfft2(torch.fft.rfft2(img3) * postfilter_pt[:, :self.N + 1])).cpu().numpy()
        return res

    def batchreconstruct_pytorch(self, img):
        assert pytorch, "No pytorch present"
        if torch.has_cuda:
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')
        nim = img.shape[0]
        r = np.mod(nim, 2 * self._nsteps)
        if r > 0:  # pad with empty frames so total number of frames is divisible by 14
            img = np.concatenate((img, np.zeros((2 * self._nsteps - r, self.N, self.N), img.dtype)))
            nim = nim + 2 * self._nsteps - r
        nimg = nim // self._nsteps
        img = torch.as_tensor(np.single(img), dtype=torch.float32, device=self.tdev)
        imf = torch.fft.rfft2(img) * torch.as_tensor(self._prefilter[:, 0:self.N // 2 + 1], device=self.tdev)

        img2 = torch.zeros((nim, 2 * self.N, 2 * self.N), dtype=torch.float32, device=self.tdev)
        bcarray = torch.zeros((self._nsteps, 2 * self.N, self.N + 1), dtype=torch.complex64, device=self.tdev)
        reconfactor_pt = torch.as_tensor(self._reconfactor, device=self.tdev)
        for i in range(0, nim, self._nsteps):
            bcarray[:, 0:self.N // 2, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, 0:self.N // 2, 0:self.N // 2 + 1]
            bcarray[:, 3 * self.N // 2:2 * self.N, 0:self.N // 2 + 1] = imf[i:i + self._nsteps, self.N // 2:self.N,
                                                                        0:self.N // 2 + 1]
            img2[i:i + self._nsteps, :, :] = torch.fft.irfft2(bcarray) * reconfactor_pt

        img3 = torch.fft.irfft(torch.fft.rfft(img2, nim, 0)[0:nimg // 2 + 1, :, :], nimg, 0)

        postfilter_pt = torch.as_tensor(self._postfilter, device=self.tdev)
        res = (torch.fft.irfft2(torch.fft.rfft2(img3) * postfilter_pt[:, :self.N + 1])).cpu().numpy()
        return res

    def empty_cache(self):
        if pytorch:
            print(f'\ttorch cuda memory reserved: {torch.cuda.memory_reserved() / 1e9} GB')
            torch.cuda.empty_cache()
            print(f'\ttorch cuda memory reserved after clearing: {torch.cuda.memory_reserved() / 1e9} GB')
        if cupy:
            print(f'\tcupy memory used: {cp._default_memory_pool.used_bytes() / 1e9} GB')
            print(f'\tcupy memory total: {cp._default_memory_pool.total_bytes() / 1e9} GB')
            cp._default_memory_pool.free_all_blocks()
            print(f'\tcupy memory used after clearing: {cp._default_memory_pool.used_bytes() / 1e9} GB')
            print(f'\tcupy memory total after clearing: {cp._default_memory_pool.total_bytes() / 1e9} GB')

    def find_phase(self, kx, ky, img, useCupy=False, useTorch=False):
        # Finds the complex correlation coefficients of the spatial frequency component at (kx, ky)
        # in the images in img.  Run after calibration with a full set to find kx and estimated band0 component
        # Combines the given images modulo the self._nsteps, and returns self._nsteps amplitude and phase values
        assert (img.shape[0] >= self._nsteps), f'number of images in find_phase should be >= {self._nsteps}'
        nimages = (img.shape[0] // self._nsteps) * self._nsteps
        (kx, ky) = (kx.squeeze(), ky.squeeze())
        if useTorch:
            img = torch.as_tensor(img, dtype=torch.float32, device=self.tdev)
            kr = torch.as_tensor(self._kr, device=self.tdev)
            otf = torch.ones((self.N, self.N), dtype=torch.float32, device=self.tdev)
            imgnsum = torch.zeros((self._nsteps, img.shape[1], img.shape[2]), dtype=torch.float32, device=self.tdev)
            xx = torch.arange(-self.N / 2 * self._dx, self.N / 2 * self._dx, self._dx, dtype=torch.float64, device=self.tdev)
            (TF, FFT, xp) = (self._tf_pytorch, torch.fft, torch)
        elif useCupy:
            img = cp.asarray(img, dtype=cp.float32)
            kr = cp.asarray(self._kr)
            otf = cp.ones((self.N, self.N), dtype=cp.float32)
            imgnsum = cp.zeros((self._nsteps, img.shape[1], img.shape[2]), dtype=cp.float32)
            xx = cp.arange(-self.N / 2 * self._dx, self.N / 2 * self._dx, self._dx, dtype=cp.float64)
            (TF, FFT, xp) = (self._tf_cupy, cp.fft, cp)
        else:
            img = np.float32(img)
            kr = self._kr
            otf = np.ones((self.N, self.N), dtype = np.float32)
            imgnsum = np.zeros((self._nsteps, img.shape[1], img.shape[2]), dtype = np.single)
            xx = np.arange(-self.N / 2 * self._dx, self.N / 2 * self._dx, self._dx, dtype=np.double)
            (TF, FFT, xp) = (self._tf, np.fft, np)

        hpf = 1.0 * (kr > self.eta / 2)
        m = kr < self.eta * 2
        otf[m] = TF(kr[m])

        hpf[m] = hpf[m] / otf[m]
        hpf[~m] = 0
        hpf = FFT.fftshift(hpf)

        imgsum = xp.sum(img[:nimages, :, :], 0) / nimages
        for i in range(self._nsteps):
            imgnsum[i, :, :] = xp.sum(img[i:nimages:self._nsteps, :, :], 0) * self._nsteps / nimages

        p = FFT.ifft2(FFT.fft2(imgnsum - imgsum) * hpf)
        p0 = FFT.ifft2(FFT.fft2(imgsum) * hpf)
        phase_shift_to_xpeak = xp.exp(1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = xp.exp(1j * ky * xx * 2 * pi * self.NA / self.wavelength)
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
            pxc0 = np.int(np.round(kx / self._dk) + self.N / 2)
            pyc0 = np.int(np.round(ky / self._dk) + self.N / 2)
            circle = plt.Circle((pxc0, pyc0), color='red', fill=False)
            ax.add_artist(circle)
            mag = 25 * self.N / 256
            if useTorch:
                ixfz, Kx, Ky = self._zoomf(p0.cpu().numpy() * p[0, :, :].cpu().numpy(), self.N, np.single(kx), np.single(ky), mag,
                                       self._dk * self.N)
            elif useCupy:
                ixfz, Kx, Ky = self._zoomf(p0.get() * p[0, :, :].get(), self.N, np.single(kx), np.single(ky), mag,
                                       self._dk * self.N)
            else:
                ixfz, Kx, Ky = self._zoomf(p0 * p[0, :, :], self.N, np.single(kx), np.single(ky), mag,
                                       self._dk * self.N)
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

    def _coarseFindCarrier(self, band0, band1, mask):
        otf_exclude_min_radius = self.eta / 2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        otf_exclude_max_radius = self.eta * 2 # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        maskbpf = (self._kr > otf_exclude_min_radius) & (self._kr < otf_exclude_max_radius)

        motf = fft.fftshift(maskbpf / (self._tfm(self._kr, maskbpf) + (1 - maskbpf) * 0.0001))

        band0_common = fft.ifft2(fft.fft2(band0) * motf)
        band1_common = fft.ifft2(fft.fft2(np.conjugate(band1)) * motf)
        # ix = band0_common * np.conjugate(band1_common)
        ix = band0_common * band1_common

        ixf = np.abs(fft.fftshift(fft.fft2(fft.fftshift(ix))))
        #self.ixf = ixf

        # pyc0, pxc0 = self._findPeak((ixf - gaussian_filter(ixf, 20)) * mask)
        pyc0, pxc0 = self._findPeak(ixf * mask)
        kx = self._dk * (pxc0 - self.N / 2)
        ky = self._dk * (pyc0 - self.N / 2)

        if self.debug:
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf, cmap = plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = plt.Circle((self.N // 2, self.N // 2), radius=2 / self._dk, color='green', fill=False)
            ax.add_artist(circle)
            circle = plt.Circle((self.N // 2, self.N // 2), radius=1.9 * self.eta / self._dk, color='cyan', fill=False)
            ax.add_artist(circle)

        return kx, ky

    def _refineCarrier(self, band0, band1, kx_in, ky_in):
        pxc0 = np.int(np.round(kx_in / self._dk) + self.N // 2)
        pyc0 = np.int(np.round(ky_in / self._dk) + self.N // 2)

        otf_exclude_min_radius = self.eta / 2
        otf_exclude_max_radius = self.eta * 2

        m = (self._kr < 2)
        otf = fft.fftshift(self._tfm(self._kr, m) + (1 - m) * 0.0001)

        otf_mask = (self._kr > otf_exclude_min_radius) & (self._kr < otf_exclude_max_radius)
        otf_mask_for_band_common_freq = fft.fftshift(
            otf_mask & scipy.ndimage.shift(otf_mask, (pyc0 - (self.N // 2 ), pxc0 - (self.N // 2 )), order=0))

        if self.debug:
            plt.figure()
            plt.title('Common mask')
            plt.imshow(fft.fftshift(otf_mask_for_band_common_freq), cmap=plt.get_cmap('gray'))

        band0_common = fft.ifft2(fft.fft2(band0) / otf * otf_mask_for_band_common_freq)

        band1_common = fft.ifft2(fft.fft2(np.conjugate(band1)) / otf * otf_mask_for_band_common_freq)

        band = band0_common * band1_common

        if self.debug:
            ixf = np.abs(fft.fftshift(fft.fft2(fft.fftshift(band))))
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf, cmap = plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = plt.Circle((self.N // 2, self.N // 2), radius=2 / self._dk, color='green', fill=False)
            ax.add_artist(circle)

        mag = 25 * self.N / 256
        ixfz, Kx, Ky = self._zoomf(band, self.N, np.single(self._k[pxc0]), np.single(self._k[pyc0]), mag , self._dk * self.N)
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

        xx = np.arange(-self.N / 2, self.N / 2, dtype=np.double) * self._dx
        phase_shift_to_xpeak = exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = exp(-1j * ky * xx * 2 * pi * self.NA / self.wavelength)

        scaling = 1 / np.sum(band0_common * np.conjugate(band0_common))
        cross_corr_result = np.sum(band0_common * band1_common * np.outer(
                        phase_shift_to_ypeak, phase_shift_to_xpeak)) * scaling

        ampl = np.abs(cross_corr_result) * 2
        phase = np.angle(cross_corr_result)
        return kx, ky, phase, ampl

    def _coarseFindCarrier_cupy(self, band0, band1, mask):
        band0 = cp.asarray(band0)
        band1 = cp.asarray(band1)
        mask = cp.asarray(mask)
        kr = cp.asarray(self._kr)

        otf_exclude_min_radius = self.eta / 2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        otf_exclude_max_radius = self.eta * 2 # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
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
        kx = self._dk * (pxc0 - self.N / 2)
        ky = self._dk * (pyc0 - self.N / 2)

        return kx.get(), ky.get()

    def _refineCarrier_cupy(self, band0, band1, kx_in, ky_in):
        band0 = cp.asarray(band0)
        band1 = cp.asarray(band1)

        pxc0 = np.int(np.round(kx_in/self._dk)+self.N//2)
        pyc0 = np.int(np.round(ky_in/self._dk)+self.N//2)

        otf_exclude_min_radius = self.eta/2
        otf_exclude_max_radius = 1.5

        # kr = cp.sqrt(cp.asarray(self._kx) ** 2 + cp.asarray(self._ky) ** 2)
        kr = cp.asarray(self._kr, dtype=np.double)
        m = (kr < 2)
        otf = cp.fft.fftshift(self._tfm_cupy(kr, m) + (1 - m)*0.0001)

        otf_mask = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)
        otf_mask_for_band_common_freq = cp.fft.fftshift(
            otf_mask & cupyx.scipy.ndimage.shift(otf_mask, (pyc0 - (self.N // 2), pxc0 - (self.N // 2)), order=0))

        band0_common = cp.fft.ifft2(cp.fft.fft2(band0) / otf * otf_mask_for_band_common_freq)
        band1_common = cp.fft.ifft2(cp.fft.fft2(cp.conjugate(band1)) / otf * otf_mask_for_band_common_freq)

        band = band0_common * band1_common

        mag = 25 * self.N / 256
        ixfz, Kx, Ky = self._zoomf_cupy(band, self.N, np.single(self._k[pxc0]), np.single(self._k[pyc0]), mag, self._dk * self.N)
        pyc, pxc = self._findPeak_cupy(abs(ixfz))

        if self.debug:
            plt.figure()
            plt.title('Zoom Find carrier')
            plt.imshow(abs(ixfz.get()))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = plt.Circle((self.N // 2, self.N // 2), radius=2 / self._dk, color='green', fill=False)
            ax.add_artist(circle)

        kx = Kx[pxc]
        ky = Ky[pyc]

        xx = cp.arange(-self.N / 2, self.N / 2, dtype=np.double) * self._dx
        phase_shift_to_xpeak = cp.exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = cp.exp(-1j * ky * xx * 2 * pi * self.NA / self.wavelength)

        scaling = 1 / cp.sum(band0_common * cp.conjugate(band0_common))

        cross_corr_result = cp.sum(band0_common * band1_common * cp.outer(
            phase_shift_to_ypeak, phase_shift_to_xpeak)) * scaling

        ampl = cp.abs(cross_corr_result) * 2
        phase = cp.angle(cross_corr_result)
        return kx.get(), ky.get(), phase.get(), ampl.get()

    def _coarseFindCarrier_pytorch(self, band0, band1, mask):
        band0 = torch.as_tensor(band0, device=self.tdev)
        band1 = torch.as_tensor(band1, device=self.tdev)
        mask = torch.as_tensor(mask, device=self.tdev)
        kr = torch.as_tensor(self._kr, device=self.tdev)

        otf_exclude_min_radius = self.eta / 2 # Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        otf_exclude_max_radius = self.eta * 2 # Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
        maskbpf = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)

        motf = torch.fft.fftshift(maskbpf / (self._tfm_pytorch(kr, maskbpf) + (~maskbpf) * 0.0001))

        band0_common = torch.fft.ifft2(torch.fft.fft2(band0) * motf)
        band1_common = torch.fft.ifft2(torch.fft.fft2(torch.conj(band1)) * motf)
        ix = band0_common * band1_common

        ixf = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(ix))))
        # self.ixf = ixf.cpu().numpy()
        pyc0, pxc0 = self._findPeak_pytorch(ixf * mask)
        kx = (self._dk * (pxc0 - self.N / 2)).cpu().numpy()
        ky = (self._dk * (pyc0 - self.N / 2)).cpu().numpy()

        if self.debug:
            plt.figure()
            plt.title('Find carrier')
            plt.imshow(ixf.cpu().numpy(), cmap=plt.get_cmap('gray'))
            ax = plt.gca()
            circle = plt.Circle((pxc0, pyc0), color = 'red', fill = False)
            ax.add_artist(circle)
            circle = plt.Circle((self.N // 2, self.N // 2), radius=2 / self._dk, color='green', fill=False)
            ax.add_artist(circle)

        return kx, ky

    def _refineCarrier_pytorch(self, band0, band1, kx_in, ky_in):
        band0 = torch.as_tensor(band0, device=self.tdev)
        band1 = torch.as_tensor(band1, device=self.tdev)

        pxc0 = np.int(np.round(kx_in/self._dk) + self.N // 2)
        pyc0 = np.int(np.round(ky_in/self._dk) + self.N // 2)

        otf_exclude_min_radius = self.eta / 2
        otf_exclude_max_radius = self.eta * 2

        # kr = cp.sqrt(cp.asarray(self._kx) ** 2 + cp.asarray(self._ky) ** 2)
        kr = torch.as_tensor(self._kr, dtype=torch.float, device=self.tdev)
        m = (kr < 2)
        otf = torch.fft.fftshift(self._tfm_pytorch(kr, m) + (~m)*0.0001)

        otf_mask = (kr > otf_exclude_min_radius) & (kr < otf_exclude_max_radius)
        shiftx = pxc0 - (self.N // 2)
        shifty = pyc0 - (self.N // 2)
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
            circle = plt.Circle((self.N // 2, self.N // 2), radius=2 / self._dk, color='green', fill=False)
            ax.add_artist(circle)

        mag = 25 * self.N / 256
        ixfz, Kx, Ky = self._zoomf_pytorch(band, self.N, np.single(self._k[pxc0]), np.single(self._k[pyc0]), mag, self._dk * self.N)
        pyc, pxc = self._findPeak_pytorch(abs(ixfz))

        if self.debug:
            plt.figure()
            plt.title('Zoon Find carrier')
            plt.imshow(abs(ixfz.cpu().numpy()))

        kx = Kx[pxc]
        ky = Ky[pyc]

        xx = torch.arange(-self.N / 2, self.N / 2, dtype=torch.float, device=self.tdev) * self._dx
        phase_shift_to_xpeak = torch.exp(-1j * kx * xx * 2 * pi * self.NA / self.wavelength)
        phase_shift_to_ypeak = torch.exp(-1j * ky * xx * 2 * pi * self.NA / self.wavelength)

        scaling = 1 / torch.sum(band0_common * torch.conj(band0_common))

        cross_corr_result = torch.sum(band0_common * band1_common * torch.outer(
            phase_shift_to_ypeak, phase_shift_to_xpeak)) * scaling

        ampl = torch.abs(cross_corr_result) * 2
        phase = torch.angle(cross_corr_result)
        return kx.cpu().numpy(), ky.cpu().numpy(), phase.cpu().numpy(), ampl.cpu().numpy()

    def _findPeak(self, in_array):
        return np.unravel_index(np.argmax(in_array, axis=None), in_array.shape)

    def _findPeak_cupy(self, in_array):
        xp = cp.get_array_module(in_array)
        return xp.unravel_index(xp.argmax(in_array, axis=None), in_array.shape)

    def _findPeak_pytorch(self, in_array):
        indx = torch.argmax(in_array, axis=None)
        return torch.div(indx, in_array.shape[0], rounding_mode='floor'), indx % in_array.shape[0]

    def _zoomf(self, in_arr, M, kx, ky, mag, kmax):
        resy = self._pyczt(in_arr, M, exp(-1j * 2 * pi / (mag * M)), exp(-1j * pi * (1 / mag - 2 * ky / kmax)))
        res = self._pyczt(resy.T, M, exp(-1j * 2 * pi / (mag * M)), exp(-1j * pi * (1 / mag - 2 * kx / kmax))).T
        kyarr = -kmax * (1 / mag - 2 * ky / kmax) / 2 + (kmax / (mag * M)) * np.arange(0, M)
        kxarr = -kmax * (1 / mag - 2 * kx / kmax) / 2 + (kmax / (mag * M)) * np.arange(0, M)
        dim = np.shape(in_arr)
        # remove phase tilt from (0,0) offset in spatial domain
        res = res * (exp(1j * kyarr * dim[0] * pi / kmax)[:, np.newaxis])
        res = res * (exp(1j * kxarr * dim[0] * pi / kmax)[np.newaxis, :])
        return res, kxarr, kyarr

    def _zoomf_cupy(self, in_arr, M, kx, ky, mag, kmax):
        resy = self._pyczt_cupy(in_arr, M, cp.exp(-1j * 2 * pi / (mag * M)),
                                cp.exp(-1j * pi * (1 / mag - 2 * ky / kmax)))
        res = self._pyczt_cupy(resy.T, M, cp.exp(-1j * 2 * pi / (mag * M)),
                               cp.exp(-1j * pi * (1 / mag - 2 * kx / kmax))).T
        kyarr = -kmax * (1 / mag - 2 * ky / kmax) / 2 + (kmax / (mag * M)) * cp.arange(0, M)
        kxarr = -kmax * (1 / mag - 2 * kx / kmax) / 2 + (kmax / (mag * M)) * cp.arange(0, M)
        dim = cupyx.scipy.sparse.csr_matrix.get_shape(in_arr)
        # remove phase tilt from (0,0) offset in spatial domain
        res = res * (cp.exp(1j * kyarr * dim[0] * pi / kmax)[:, cp.newaxis])
        res = res * (cp.exp(1j * kxarr * dim[0] * pi / kmax)[cp.newaxis, :])
        return res, kxarr, kyarr

    def _zoomf_pytorch(self, in_arr, M, kx, ky, mag, kmax):
        resy = self._pyczt_pytorch(in_arr, M, torch.exp(torch.tensor(-1j * 2 * pi / (mag * M), device=self.tdev)),
                                torch.exp(torch.tensor(-1j * pi * (1 / mag - 2 * ky / kmax), device=self.tdev)))
        res = self._pyczt_pytorch(resy.T, M, torch.exp(torch.tensor(-1j * 2 * pi / (mag * M), device=self.tdev)),
                               torch.exp(torch.tensor(-1j * pi * (1 / mag - 2 * kx / kmax), device=self.tdev))).T
        kyarr = -kmax * (1 / mag - 2 * ky / kmax) / 2 + (kmax / (mag * M)) * torch.arange(0, M, device=self.tdev)
        kxarr = -kmax * (1 / mag - 2 * kx / kmax) / 2 + (kmax / (mag * M)) * torch.arange(0, M, device=self.tdev)
        dim = in_arr.shape
        # remove phase tilt from (0,0) offset in spatial domain
        res = res * torch.unsqueeze(torch.exp(1j * kyarr * dim[0] * pi / kmax), 1)
        res = res * torch.unsqueeze(torch.exp(1j * kxarr * dim[0] * pi / kmax), 0)
        return res, kxarr, kyarr

    def _att(self, kr):
        atf = (1 - self.beta * exp(-kr ** 2 / (2 * self.alpha ** 2)))
        return atf

    def _attm(self, kr, mask):
        atf = np.zeros_like(kr)
        atf[mask] = self._att(kr[mask])
        return atf

    def _tf(self, kr, a_type=None):
        otf = np.zeros_like(kr)
        m = kr < 2.0
        otf[m] = (2 / pi * (arccos(kr[m] / 2) - kr[m] / 2 * sqrt(1 - kr[m] ** 2 / 4)))
        if a_type is None:
            a_type = self.a_type
        if a_type == 'exp':
            return otf * (self.a ** (kr / 2))
        elif a_type == 'sph':
            return otf * (1 - (1 - self.a) * (1 - (kr - 1) ** 2))
        else:
            return otf

    def _tf_cupy(self, kr, a_type=None):
        xp = cp.get_array_module(kr)
        otf = xp.zeros(kr.shape, dtype=xp.float)
        m = kr < 2.0
        otf[m] = (2 / pi * (xp.arccos(kr[m] / 2) - kr[m] / 2 * xp.sqrt(1 - kr[m] ** 2 / 4)))
        if a_type is None:
            a_type = self.a_type
        if a_type == 'exp':
            return otf * (self.a ** (kr / 2))
        elif a_type == 'sph':
            return otf * (1 - (1 - self.a) * (1 - (kr - 1) ** 2))
        else:
            return otf

    def _tf_pytorch(self, kr, a_type=None):
        otf = torch.zeros_like(kr)
        m = kr < 2.0
        otf[m] = (2 / pi * (torch.arccos(kr[m] / 2) - kr[m] / 2 * torch.sqrt(1 - kr[m] ** 2 / 4)))
        if a_type is None:
            a_type = self.a_type
        if a_type == 'exp':
            return otf * (self.a ** (kr / 2))
        elif a_type == 'sph':
            return otf * (1 - (1 - self.a) * (1 - (kr - 1) ** 2))
        else:
            return otf

    def _tfm(self, kr, mask):
        otf = np.zeros_like(kr)
        otf[mask] = self._tf(kr[mask])
        return otf

    def _tfm_cupy(self, kr, mask):
        xp = cp.get_array_module(kr)
        otf = xp.zeros_like(kr)
        otf[mask] = self._tf_cupy(kr[mask])
        return otf

    def _tfm_pytorch(self, kr, mask):
        otf = torch.zeros_like(kr)
        otf[mask] = self._tf_pytorch(kr[mask])
        return otf

    def _pyczt(self, x, k=None, w=None, a=None):
        # Chirp z-transform ported from Matlab implementation (see comment below)
        # By Mark Neil Apr 2020
        # %CZT  Chirp z-transform.
        # %   G = CZT(X, M, W, A) is the M-element z-transform of sequence X,
        # %   where M, W and A are scalars which specify the contour in the z-plane
        # %   on which the z-transform is computed.  M is the length of the transform,
        # %   W is the complex ratio between points on the contour, and A is the
        # %   complex starting point.  More explicitly, the contour in the z-plane
        # %   (a spiral or "chirp" contour) is described by
        # %       z = A * W.^(-(0:M-1))
        # %
        # %   The parameters M, W, and A are optional; their default values are
        # %   M = length(X), W = exp(-j*2*pi/M), and A = 1.  These defaults
        # %   cause CZT to return the z-transform of X at equally spaced points
        # %   around the unit circle, equivalent to FFT(X).
        # %
        # %   If X is a matrix, the chirp z-transform operation is applied to each
        # %   column.
        # %
        # %   See also FFT, FREQZ.
        #
        # %   Author(s): C. Denham, 1990.
        # %   	   J. McClellan, 7-25-90, revised
        # %   	   C. Denham, 8-15-90, revised
        # %   	   T. Krauss, 2-16-93, updated help
        # %   Copyright 1988-2002 The MathWorks, Inc.
        # %       $Revision: 1.7.4.1 $  $Date: 2007/12/14 15:04:15 $
        #
        # %   References:
        # %     [1] Oppenheim, A.V. & R.W. Schafer, Discrete-Time Signal
        # %         Processing,  Prentice-Hall, pp. 623-628, 1989.
        # %     [2] Rabiner, L.R. and B. Gold, Theory and Application of
        # %         Digital Signal Processing, Prentice-Hall, Englewood
        # %         Cliffs, New Jersey, pp. 393-399, 1975.

        olddim = x.ndim

        if olddim == 1:
            x = x[:, np.newaxis]

        (m, n) = x.shape
        oldm = m

        if m == 1:
            x = x.transpose()
            (m, n) = x.shape

        if k is None:
            k = len(x)
        if w is None:
            w = exp(-1j * 2 * pi / k)
        if a is None:
            a = 1.
        # %------- Length for power-of-two fft.

        nfft = int(2 ** np.ceil(log2(abs(m + k - 1))))

        # %------- Premultiply data.

        kk = np.arange(-m + 1, max(k, m))[:, np.newaxis]
        kk2 = (kk ** 2) / 2
        ww = w ** kk2  # <----- Chirp filter is 1./ww
        nn = np.arange(0, m)[:, np.newaxis]
        aa = a ** (-nn)
        aa = aa * ww[m + nn - 1, 0]
        # y = (x * aa)
        y = (x * aa).astype(np.complex64)
        # print(y.dtype)
        # %------- Fast convolution via FFT.

        fy = fft.fft(y, nfft, axis=0)
        fv = fft.fft(1 / ww[0: k - 1 + m], nfft, axis=0)  # <----- Chirp filter.
        fy = fy * fv
        g = fft.ifft(fy, axis=0)

        # %------- Final multiply.

        g = g[m - 1:m + k - 1, :] * ww[m - 1:m + k - 1]
        if oldm == 1:
            g = g.transpose()

        if olddim == 1:
            g = g.squeeze()

        return g

    def _pyczt_cupy(self, x, k=None, w=None, a=None):
        olddim = x.ndim

        if olddim == 1:
            x = x[:, cp.newaxis]

        (m, n) = x.shape
        oldm = m

        if m == 1:
            x = x.transpose()
            (m, n) = x.shape

        if k is None:
            k = len(x)
        if w is None:
            w = cp.exp(-1j * 2 * pi / k)
        if a is None:
            a = 1.

        # %------- Length for power-of-two fft.

        nfft = int(2 ** cp.ceil(cp.log2(abs(m + k - 1))))

        # %------- Premultiply data.

        kk = cp.arange(-m + 1, max(k, m))[:, cp.newaxis]
        kk2 = (kk ** 2) / 2
        ww = w ** kk2  # <----- Chirp filter is 1./ww
        nn = cp.arange(0, m)[:, cp.newaxis]
        aa = a ** (-nn)
        aa = aa * ww[m + nn - 1, 0]
        y = (x * aa).astype(np.complex64)

        # %------- Fast convolution via FFT.

        fy = cp.fft.fft(y, nfft, axis=0)
        fv = cp.fft.fft(1 / ww[0: k - 1 + m], nfft, axis=0)  # <----- Chirp filter.
        fy = fy * fv
        g = cp.fft.ifft(fy, axis=0)

        # %------- Final multiply.

        g = g[m - 1:m + k - 1, :] * ww[m - 1:m + k - 1]

        if oldm == 1:
            g = g.transpose()

        if olddim == 1:
            g = g.squeeze()

        return g

    def _pyczt_pytorch(self, x, k=None, w=None, a=None):
        olddim = x.ndim

        if olddim == 1:
            x = torch.unsqueeze(x, 1)

        (m, n) = x.shape
        oldm = m

        if m == 1:
            x = x.transpose()
            (m, n) = x.shape

        if k is None:
            k = len(x)
        if w is None:
            w = torch.exp(-1j * 2 * pi / k)
        if a is None:
            a = 1.

        # %------- Length for power-of-two fft.

        nfft = int(2 ** np.ceil(np.log2(abs(m + k - 1))))

        # %------- Premultiply data.

        kk = torch.unsqueeze(torch.arange(-m + 1, max(k, m), device=self.tdev), 1)
        kk2 = (kk ** 2) / 2
        ww = w ** kk2  # <----- Chirp filter is 1./ww
        nn = torch.unsqueeze(torch.arange(0, m, device=self.tdev), 1)
        aa = a ** (-nn)
        aa = aa * ww[m + nn - 1, 0]
        y = x * aa

        # %------- Fast convolution via FFT.

        fy = torch.fft.fft(y, nfft, axis=0)
        fv = torch.fft.fft(1 / ww[0: k - 1 + m], nfft, axis=0)  # <----- Chirp filter.
        fy = fy * fv
        g = torch.fft.ifft(fy, axis=0)

        # %------- Final multiply.

        g = g[m - 1:m + k - 1, :] * ww[m - 1:m + k - 1]

        if oldm == 1:
            g = g.transpose()

        if olddim == 1:
            g = g.squeeze()

        return g
