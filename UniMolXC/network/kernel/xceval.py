'''
the interface to libxc for evaluating the exchange-correlation energies
and all its possible derivatives.
'''
import os
import re
import unittest

import numpy as np
import pylibxc as libxc
from scipy.integrate import simpson

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

class TestXCEval(unittest.TestCase):
    
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        testfiles = os.path.dirname(here)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.join(testfiles, 'testfiles')
        self.frho_pbe = os.path.join(testfiles, 'pbe-rho.cube')
        self.frho_m06l = os.path.join(testfiles, 'm06l-rho.cube')
        self.ftau_m06l = os.path.join(testfiles, 'm06l-tau.cube')
    
    def test_validate_xc_notation(self):
        
        testxc = ['LDA_X', 'GGA_C_PBE', 'HYB_GGA_XC_B3LYP', 
                  'MGGA_K_TPSS', 'MGGA_HYB_XC_M06']
        res = [validate_xc_notation(xc) for xc in testxc]
        self.assertTrue(all(res))

    def test_calculate_exc(self):
        cube = read(self.frho_pbe)
        rho, dv, nelec = cube['data'], np.linalg.det(cube['R']), np.sum(cube['chg'])
        rho = np.array(rho, dtype=np.float64) * dv # rho should be normalized to nelec
        self.assertAlmostEqual(np.sum(rho), nelec, delta=1e-4)
        
        nxyz, = rho.shape
        
        myexch : dict[str, np.ndarray] = calculate(
            xc='GGA_X_PBE', 
            spin=1, 
            densities={'rho': rho, 'sigma': np.zeros_like(rho)}, 
            do_exc=True, 
            do_vxc=False)
        
        self.assertIsNotNone(myexch)
        self.assertIn('zk', myexch) # do not know why zk stands for exc...
        self.assertIsNotNone(myexch['zk'])
        self.assertIsInstance(myexch['zk'], np.ndarray)
        self.assertEqual(myexch['zk'].shape, (nxyz, 1))

        mycorr = calculate(
            xc='GGA_C_PBE', 
            spin=1, 
            densities={'rho': rho, 'sigma': np.zeros_like(rho)}, 
            do_exc=True, 
            do_vxc=False)

        Exc : np.ndarray = np.dot(rho, myexch['zk'] + mycorr['zk'])
        self.assertEqual(Exc.shape, (1,))
        Exc = Exc[0]
        self.assertIsInstance(Exc, np.float64)
        self.assertAlmostEqual(Exc, -65.7266949583, delta=1e-5)

        rho = read(self.frho_m06l)['data']
        rho = np.array(rho, dtype=np.float64) * dv
        tau = read(self.ftau_m06l)['data']
        tau = np.array(tau, dtype=np.float64) * dv
        myexch = calculate(
            xc='MGGA_X_M06_L',
            spin=1,
            densities={'rho': rho, 'tau': tau, 'sigma': np.zeros_like(rho)},
            do_exc=True,
            do_vxc=False
        )
        self.assertIsNotNone(myexch)
        self.assertIn('zk', myexch)
        self.assertIsNotNone(myexch['zk'])
        self.assertIsInstance(myexch['zk'], np.ndarray)
        self.assertEqual(myexch['zk'].shape, (nxyz, 1))
        
        Ex = np.dot(rho, myexch['zk'])
        self.assertEqual(Ex.shape, (1,))
        Ex = Ex[0]
        self.assertIsInstance(Ex, np.float64)
        self.assertLess(Ex, 0.0)

if __name__ == '__main__':
    unittest.main()
