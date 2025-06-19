'''
Generate the DeePMD-kit representation of a structure with
the 'descriptor' of DeePMD model.
'''
# in-built modules
import os
import unittest

# third-party modules
import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell
try:
    from deepmd.infer.deep_pot import DeepPot
except ImportError:
    raise ImportError('DeePMD-kit is not installed. '
        'See https://docs.deepmodeling.com/projects/deepmd/'
        'en/stable/getting-started/install.html#install-with-conda')

# local modules
from UniMolXC.abacus.control import AbacusJob

def generate_from_abacus(job,
                         dpmodel,
                         head=None):
    '''
    With one ABACUS jobdir, generate the DeePMD representation
    of the structure.
    
    Parameters
    ----------
    job : str or AbacusJob
        The path to the ABACUS job directory or instance of
        `UniMolXC.abacus.control.AbacusJob`.
    dpmodel : str
        The path to the DeePMD model file, or the instance of
        `deepmd.infer.deep_pot.DeepPot`.
    head : str, optional
        deprecated, the head of the DeePMD model.
    
    Returns
    -------
    np.ndarray
        The DeePMD representation of the structure, will be in
        shape of nframes x n_atoms x ?.
    '''
    # simple sanity check
    assert isinstance(job, (str, AbacusJob))
    if isinstance(dpmodel, str):
        assert os.path.exists(dpmodel), f'{dpmodel} does not exist'
        dpmodel = DeepPot(dpmodel)
    else:
        assert isinstance(dpmodel, DeepPot)

    # convert the structure to the DPData format
    data = AbacusJob(job).to_deepmd() if isinstance(job, str) else job.to_deepmd()
    
    type_map_model = dpmodel.get_type_map()
    atomtype = np.array([type_map_model.index(data['atom_names'][it_at])
                         for it_at in data['atom_types']])
    # the DeePMD model may crash for left-handed cells, so we convert
    # the cell parameters to the right-handed ones.
    cells = np.array([cellpar_to_cell(cell_to_cellpar(c)).flatten() \
                      if np.linalg.det(c) < 0 else c
                      for c in data['cells']]).reshape(-1, 9)
    return dpmodel.eval_descriptor(
        coords=data['coords'],
        cells=cells,
        atom_types=atomtype
    )

class TestAbacusToDeePMDRepr(unittest.TestCase):
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
    
    # @unittest.skip('Skip this test due to the error raised by the DeePMD-kit')
    def test_generate_from_abacus(self):
        try:
            import deepmd
        except ImportError:
            self.skipTest('DeePMD-kit is not installed. '
                'See https://docs.deepmodeling.com/projects/deepmd/en/'
                'stable/getting-started/install.html#install-with-conda')

        fmodel = os.path.join(self.testfiles, 'dpa3-2p4-7m-mptraj.pth')
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        # from an finished ABACUS job
        result1 = generate_from_abacus(jobdir, fmodel)
        nframe, nat, ndim = result1.shape
        self.assertEqual(nframe, 1)
        self.assertEqual(nat, 2)
        self.assertEqual(ndim, 448)
        
        # from an AbacusJob instance
        job = AbacusJob(jobdir)
        result2 = generate_from_abacus(job, fmodel)
        self.assertTrue(np.array_equal(result1, result2))
        
        # from a DeepPot instance
        dpmodel = DeepPot(fmodel)
        result3 = generate_from_abacus(job, dpmodel)
        self.assertTrue(np.array_equal(result1, result3))

if __name__ == '__main__':
    unittest.main()