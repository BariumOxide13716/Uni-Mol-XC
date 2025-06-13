'''
the interface to libxc for evaluating the exchange-correlation energies
and all its possible derivatives.
'''
import os
import re
import unittest
from typing import Optional, Tuple

import numpy as np
import pylibxc as libxc

from UniMolXC.utility.cubeio import read

def validate_xc_notation(xc: str):
    '''
    validate whether the xc functional notation is valid. The only
    legal format of notation is:
    
    (LDA|GGA|HYB_GGA|MGGA|MGGA_HYB)_(X|C|XC|K)(_(.*))?
    
    Parameters
    ----------
    xc : str
    
    Returns
    -------
    dict
    A dictionary with the following keys:
    - 'type': the type of the functional (LDA, GGA, HYB_GGA, MGGA, MGGA_HYB)
    - 'kind': the kind of the functional (X, C, XC, K)
    - 'name': the name of the functional (optional, can be None)
    '''
    m = re.match(r'^(LDA|GGA|HYB_GGA|MGGA|MGGA_HYB)_(X|C|XC|K)(_(.*))?$', xc)
    if m is None:
        return {}
    else:
        return {
            'type': m.group(1),
            'kind': m.group(2),
            'name': m.group(4) if m.group(4) is not None else None
        }

def calculate(xc: str, spin: int|str, densities: dict, **kwargs):
    '''
    calculate the exc, vxc, ... on given densities (rho, sigma, tau, etc.)
    
    Parameters
    ----------
    xc : str
        the xc functional notation
    spin : int|str
        the spin multiplicity, e.g. 1 for non-spin-polarized, 2 for spin-polarized
        or 'unpolarized' for non-spin-polarized, 'polarized' for spin-polarized
    densities : dict
        a dictionary with the following keys:
        - 'rho': the density
        - 'sigma': the spin density (optional)
        - 'tau': the kinetic energy density (optional)
        - 'laplacian': the laplacian of the density (optional)
    kwargs : dict
        the type of calculation to perform, e.g.:
        - 'do_exc': bool, calculate the exchange-correlation energy
        - 'do_vxc': bool, calculate the exchange-correlation potential
        - 'do_fxc': bool
        - 'do_kxc': bool
        - 'do_lxc': bool
    '''
    # sanity checks
    xc_info = validate_xc_notation(xc)
    assert xc_info, \
        f'Invalid xc functional notation: {xc}'
    assert isinstance(spin, (int, str)), \
        f'Spin must be an integer or a string, got {type(spin)}'
    if isinstance(spin, str):
        assert spin.lower() in ['unpolarized', 'polarized'], \
            f'Spin must be "unpolarized" or "polarized", got {spin}'
            
    spin = 1 if spin in ['unpolarized', 1] else 2
    assert 'sigma' in densities or spin == 1, \
        'Spin-polarized calculation requires sigma (spin density) to be provided'
    # otherwise, auto set sigma to zero if it is functional higher than LDA
    if xc_info['type'] != 'LDA':
        densities['sigma'] = densities.get(
            'sigma', np.zeros_like(densities['rho']))
    
    myxc = libxc.LibXCFunctional(xc.lower(), spin)
    
    # workspace query
    out = {}
    myxc.compute(inp=densities, output=out, **kwargs) 
    
    # compute the exchange-correlation energy
    myxc.compute(inp=densities, output=out, **kwargs)
    return out

def calculate_density_contracted_gradients(
    grhox_up: np.ndarray, 
    grhoy_up: np.ndarray, 
    grhoz_up: np.ndarray,
    grhox_dw: Optional[np.ndarray]=None,
    grhoy_dw: Optional[np.ndarray]=None,
    grhoz_dw: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    calculate the "sigma", which is the so-called "contracted 
    gradients of density" with given gradients of the density. 
    For the definition, please refer to the Manual of LibXC:
    https://libxc.gitlab.io/manual/libxc-5.1.x/

    Set ABACUS parameter `out_chg` to 3 to output both the charge
    densities and the contracted gradients of densities.
    
    Parameters
    ----------
    grhox_up : np.ndarray
        the gradient of the density (of the first or the "alpha" spin
        channel) in x direction, whose dimension should be (nxyz,), 
        where nxyz is the number of grid points
    grhoy_up : np.ndarray
        similar with `grhox_up`, but in y direction
    grhoz_up : np.ndarray
        similar with `grhox_up`, but in z direction
    grhox_dw : np.ndarray, optional
        similar with `grhox_up`, but for the second or the "beta" spin.
        By default None, which means the system is non-spin-polarized.
    grhoy_dw : np.ndarray, optional
        similar with `grhox_dw`, but in y direction, by default None
    grhoz_dw : np.ndarray, optional
        similar with `grhox_dw`, but in z direction, by default None
    
    Returns
    -------
    tuple
        a tuple of three numpy arrays, as sigma(0), sigma(1), sigma(2).
        In which the sigma(0) is the inner product of the gradient of
        the alpha spin density, sigma(1) is the one between the alpha
        and beta spin densities, and sigma(2) is the one between the
        beta spin densities.
    '''
    assert grhox_up.shape == grhoy_up.shape == grhoz_up.shape, \
        'The gradients of density must have the same shape'
    spin_polarized = all(g is not None for g in [grhox_dw, grhoy_dw, grhoz_dw])
    assert all(g is None for g in [grhox_dw, grhoy_dw, grhoz_dw]) or spin_polarized, \
        'Either all gradients of the beta spin density are provided, or none of them'
    
    nxyz, = grhox_up.shape
    
    grho_up = np.stack([grhox_up, grhoy_up, grhoz_up], axis=-1)
    assert grho_up.shape == (nxyz, 3)
    sigma_aa = np.einsum('ij,ij->i', grho_up, grho_up)
    if not spin_polarized:
        return sigma_aa, np.zeros_like(sigma_aa), np.zeros_like(sigma_aa)
    grho_dw: np.ndarray = np.stack([grhox_dw, grhoy_dw, grhoz_dw], axis=-1)
    assert grho_dw.shape == (nxyz, 3)
    sigma_ab = np.einsum('ij,ij->i', grho_up, grho_dw)
    sigma_bb = np.einsum('ij,ij->i', grho_dw, grho_dw)
    return sigma_aa, sigma_ab, sigma_bb

class TestXCEval(unittest.TestCase):
    
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        testfiles = os.path.dirname(here)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.join(testfiles, 'testfiles')
    
    def test_validate_xc_notation(self):
        
        testxc = ['LDA_X', 'GGA_C_PBE', 'HYB_GGA_XC_B3LYP', 
                  'MGGA_K_TPSS', 'MGGA_HYB_XC_M06']
        res = [validate_xc_notation(xc) for xc in testxc]
        self.assertTrue(all(res))

    def test_calculate_density_contracted_gradients(self):
        grhox_up = np.array([1.0, 2.0, 3.0])
        grhoy_up = np.array([4.0, 5.0, 6.0])
        grhoz_up = np.array([7.0, 8.0, 9.0])
        
        sigma_aa, sigma_ab, sigma_bb = calculate_density_contracted_gradients(
            grhox_up, grhoy_up, grhoz_up)
        
        self.assertIsInstance(sigma_aa, np.ndarray)
        self.assertIsInstance(sigma_ab, np.ndarray)
        self.assertIsInstance(sigma_bb, np.ndarray)
        self.assertEqual(sigma_aa.shape, (3,))
        self.assertEqual(sigma_ab.shape, (3,))
        self.assertEqual(sigma_bb.shape, (3,))
        
        # check the values
        expected_sigma_aa = np.array([1**2+4**2+7**2,
                                      2**2+5**2+8**2,
                                      3**2+6**2+9**2])
        expected_sigma_ab = np.zeros_like(expected_sigma_aa)
        expected_sigma_bb = np.zeros_like(expected_sigma_aa)
        self.assertTrue(np.allclose(sigma_aa, expected_sigma_aa))
        self.assertTrue(np.allclose(sigma_ab, expected_sigma_ab))
        self.assertTrue(np.allclose(sigma_bb, expected_sigma_bb))
        
        grhox_dw = -grhox_up
        grhoy_dw = -grhoy_up
        grhoz_dw = -grhoz_up
        sigma_aa, sigma_ab, sigma_bb = calculate_density_contracted_gradients(
            grhox_up, grhoy_up, grhoz_up,
            grhox_dw, grhoy_dw, grhoz_dw)
        self.assertIsInstance(sigma_aa, np.ndarray)
        self.assertIsInstance(sigma_ab, np.ndarray)
        self.assertIsInstance(sigma_bb, np.ndarray)
        self.assertEqual(sigma_aa.shape, (3,))
        self.assertEqual(sigma_ab.shape, (3,))
        self.assertEqual(sigma_bb.shape, (3,))
        # check the values
        self.assertTrue(np.allclose(sigma_aa, expected_sigma_aa))
        self.assertTrue(np.allclose(sigma_ab, -expected_sigma_aa))
        self.assertTrue(np.allclose(sigma_bb, expected_sigma_aa))

    def test_si2_calculate_lda_exc(self):
        '''the case that sigma is not needed'''
        # read and basic assertion
        outdir = os.path.join(self.testfiles, 'pylibxc-lda')
        frho = os.path.join(outdir, 'chgs1.cube')
        myrho = read(frho)
        rho, dv, nelec = myrho['data'], \
            abs(np.linalg.det(myrho['R'])), np.sum(myrho['chg'])
        rho = np.array(rho, dtype=np.float64) * dv
        self.assertAlmostEqual(np.sum(rho), nelec, delta=1e-4)
        
        # calculate the exchange-correlation energy of LDA
        # exchange
        myexch : dict[str, np.ndarray] = calculate(
            xc='LDA_X',
            spin=1,
            densities={'rho': rho},
            do_exc=True,
            do_vxc=False)
        self.assertIsNotNone(myexch)
        self.assertIn('zk', myexch) # do not know why zk stands for exc...
        self.assertIsNotNone(myexch['zk'])
        self.assertIsInstance(myexch['zk'], np.ndarray)
        nxyz, = rho.shape
        self.assertEqual(myexch['zk'].shape, (nxyz, 1))
        
        # correlation
        mycorr = calculate(
            xc='LDA_C_PZ',
            spin=1,
            densities={'rho': rho},
            do_exc=True,
            do_vxc=False)
        # calculate the exchange-correlation energy
        Exc : np.ndarray = np.dot(rho, myexch['zk'] + mycorr['zk'])
        self.assertEqual(Exc.shape, (1,))
        Exc = Exc[0] * 27.2 # convert to eV
        self.assertIsInstance(Exc, np.float64)
        self.assertAlmostEqual(Exc, -65.6864997529, delta=1e-3)

    @unittest.skip('I will first correct the LDA case')
    def test_si1_calculate_pbe_exc(self):
        outdir = os.path.join(self.testfiles, 'pylibxc-pbe-1')
        frho = os.path.join(outdir, 'chgs1.cube')
        fgrhox, fgrhoy, fgrhoz = [os.path.join(outdir, f'chgs1_grad{i}.cube')
                                  for i in ['x', 'y', 'z']]
        myrho = read(frho)
        rho, dv, nelec = myrho['data'], \
            abs(np.linalg.det(myrho['R'])), np.sum(myrho['chg'])
        rho = np.array(rho, dtype=np.float64) * dv
        self.assertAlmostEqual(np.sum(rho), nelec, delta=1e-4)
        sigma_aa, sigma_ab, sigma_bb = calculate_density_contracted_gradients(
            read(fgrhox)['data'], read(fgrhoy)['data'], read(fgrhoz)['data'])
        # check shape
        self.assertTrue(all(isinstance(s, np.ndarray)
                            for s in [sigma_aa, sigma_ab, sigma_bb]))
        self.assertTrue(all(s.shape == rho.shape
                            for s in [sigma_aa, sigma_ab, sigma_bb]))
        # sigma 1 is valid: not all zeros
        self.assertFalse(np.allclose(sigma_aa, np.zeros_like(sigma_aa)))
        
        # calculate the exchange-correlation energy of PBE
        # exchange
        myexch : dict[str, np.ndarray] = calculate(
            xc='GGA_X_PBE',
            spin=1,
            densities={'rho': rho, 'sigma': sigma_aa},
            do_exc=True,
            do_vxc=False)
        self.assertIsNotNone(myexch)
        self.assertIn('zk', myexch) # do not know why zk stands for exc...
        self.assertIsNotNone(myexch['zk'])
        self.assertIsInstance(myexch['zk'], np.ndarray)
        nxyz, = rho.shape
        self.assertEqual(myexch['zk'].shape, (nxyz, 1))
        # correlation
        mycorr = calculate(
            xc='GGA_C_PBE',
            spin=1,
            densities={'rho': rho, 'sigma': sigma_aa},
            do_exc=True,
            do_vxc=False)
        # calculate the exchange-correlation energy
        Exc : np.ndarray = np.dot(rho, myexch['zk'] + mycorr['zk'])
        self.assertEqual(Exc.shape, (1,))
        Exc = Exc[0] * 27.2 # convert to eV
        self.assertIsInstance(Exc, np.float64)
        self.assertAlmostEqual(Exc, -27.3875180799, delta=1e-5)

    @unittest.skip('I will first correct the LDA case')
    def test_si2_calculate_pbe_exc(self):
        outdir = os.path.join(self.testfiles, 'pylibxc-pbe-2')
        frho = os.path.join(outdir, 'chgs1.cube')
        fgrhox, fgrhoy, fgrhoz = [os.path.join(outdir, f'chgs1_grad{i}.cube')
                                  for i in ['x', 'y', 'z']]
        myrho = read(frho)
        rho, dv, nelec = myrho['data'], \
            abs(np.linalg.det(myrho['R'])), np.sum(myrho['chg'])
        rho = np.array(rho, dtype=np.float64) * dv
        self.assertAlmostEqual(np.sum(rho), nelec, delta=1e-4)
        
        sigma_aa, sigma_ab, sigma_bb = calculate_density_contracted_gradients(
            read(fgrhox)['data'], read(fgrhoy)['data'], read(fgrhoz)['data'])

        # check shape
        self.assertTrue(all(isinstance(s, np.ndarray) 
                            for s in [sigma_aa, sigma_ab, sigma_bb]))
        self.assertTrue(all(s.shape == rho.shape 
                            for s in [sigma_aa, sigma_ab, sigma_bb]))
        
        # sigma 1 is valid: not all zeros
        self.assertFalse(np.allclose(sigma_aa, np.zeros_like(sigma_aa)))
        
        # calculate the exchange-correlation energy of PBE
        # exchange
        myexch : dict[str, np.ndarray] = calculate(
            xc='GGA_X_PBE', 
            spin=1, 
            densities={'rho': rho, 'sigma': sigma_aa}, 
            do_exc=True, 
            do_vxc=False)
        self.assertIsNotNone(myexch)
        self.assertIn('zk', myexch) # do not know why zk stands for exc...
        self.assertIsNotNone(myexch['zk'])
        self.assertIsInstance(myexch['zk'], np.ndarray)
        nxyz, = rho.shape
        self.assertEqual(myexch['zk'].shape, (nxyz, 1))
        # correlation
        mycorr = calculate(
            xc='GGA_C_PBE', 
            spin=1, 
            densities={'rho': rho, 'sigma': sigma_aa}, 
            do_exc=True, 
            do_vxc=False)
        # calculate the exchange-correlation energy
        Exc : np.ndarray = np.dot(rho, myexch['zk'] + mycorr['zk'])
        self.assertEqual(Exc.shape, (1,))
        Exc = Exc[0] * 27.2 # convert to eV
        self.assertIsInstance(Exc, np.float64)
        self.assertAlmostEqual(Exc, -65.6864997529, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
