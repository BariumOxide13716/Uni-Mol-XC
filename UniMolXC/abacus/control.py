'''
this script places utility functions for indexing the abacus
jobdir
'''
# built-in modules
import os
import re
import unittest
import uuid
import shutil
import json
import subprocess

# third-party modules
import numpy as np

# local modules
from UniMolXC.abacus.inputio import read as read_dftparam
from UniMolXC.abacus.inputio import write as write_dftparam
from UniMolXC.abacus.struio import read_stru as read_stru_
from UniMolXC.abacus.struio import write_stru
from UniMolXC.abacus.kptio import read as read_kpt
from UniMolXC.abacus.kptio import write as write_kpt
from UniMolXC.abacus.logio import read_energy
from UniMolXC.utility.units import convert_length_unit
# from UniMolXC.abacus.remote import submit

class AbacusJob:
        
    def build_derived(self, 
                      jobdir, 
                      dftparam=None, 
                      stru=None, 
                      kpt=None, 
                      instatiate=False,
                      log=True):
        '''build a derived AbacusJob instance with several parameters
        changed
        
        Parameters
        ----------
        jobdir : str
            the jobdir of the derived job
        dftparam : dict
            the dftparam of the derived job
        stru : dict
            the structure of the derived job
        kpt : dict
            the kpt of the derived job
        instatiate : bool
            whether to instantiate the derived job or not

        Returns
        -------
        AbacusJob
            the derived AbacusJob instance
        '''
        cwd = os.getcwd()
        
        os.makedirs(jobdir, exist_ok=False) # do not allow to overwrite
        os.chdir(jobdir)
        
        # INPUT
        dftparam = dftparam or {}
        dftparam = self.input|dftparam
        # remember to adjust the pseudo_dir and orbital_dir
        dftparam.update({'pseudo_dir': os.path.abspath(
            self.input.get('pseudo_dir', self.path))})
        if self.input.get('basis_type', 'pw') == 'lcao':
            dftparam.update({'orbital_dir': os.path.abspath(
                self.input.get('orbital_dir', self.path))})
        write_dftparam(dftparam, 'INPUT')
        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                'be called with `cache=True` before calling '
                'this function.')
        
        # STRU
        stru = stru or {}
        stru = self.stru|stru
        write_stru('.', stru)

        # KPT
        if self.kpt is None \
            and all([x not in dftparam for x in ['kspacing', 'gamma_only']]):
            raise RuntimeError('the function `read_kpt` should '
                    'be called with `cache=True` before calling '
                    'this function.')
        if all([x not in dftparam for x in ['kspacing', 'gamma_only']]):
            kpt = kpt or {}
            kpt = self.kpt|kpt
            write_kpt(kpt, 'KPT')
        # else:
        #   otherwise, we do not need to write KPT file
        
        if log:
            with open('XCPNTraine-param-update.json', 'w') as f:
                json.dump({'input': dftparam,
                           'stru': stru,
                           'kpt': kpt}, f, indent=4)
        
        # finally return to the original directory
        os.chdir(cwd)
        
        if instatiate:
            return AbacusJob(jobdir)
    
    def init_as_unfinished(self, jobdir):
        '''initialize the AbacusJob instance as a raw jobdir'''
        # INPUT
        fn_input = os.path.join(jobdir, 'INPUT')
        if not os.path.exists(fn_input):
            raise FileNotFoundError('`INPUT` file not found in the job directory.')
        self.fn_input = fn_input
        self.input = read_dftparam(self.fn_input)
        
        # STRU
        fn_stru = os.path.join(jobdir, self.input.get('stru_file', 'STRU'))
        if not os.path.exists(fn_stru):
            raise FileNotFoundError('`STRU` file not found in the job directory.')
        self.fn_stru = fn_stru
        self.stru = None
        
        # KPT
        fn_kpt = os.path.join(jobdir, self.input.get('kpoint_file', 'KPT'))
        if not os.path.exists(fn_kpt) and\
            self.input.get('kspacing') is None and\
                self.input.get('gamma_only') is None:
            raise FileNotFoundError('`KPT` file not found in the job directory.')
        self.fn_kpt = fn_kpt
        self.kpt = None
        
        # after the checking above, it can be a real ABACUS jobdir
        self.path = jobdir

    def init_as_finished(self, jobdir):
        '''initialize the AbacusJob instance as a finished jobdir'''
        # OUT
        outdir = os.path.join(jobdir, f'OUT.{self.input.get("suffix", "ABACUS")}')
        if not os.path.exists(outdir):
            raise FileNotFoundError('`OUT` file not found in the job directory.')
        self.outdir = outdir
        
        # running log
        flog = os.path.join(self.outdir, 
                            f'running_{self.input.get("calculation", "scf")}.log')
        if not os.path.exists(flog):
            raise FileNotFoundError('`running.log` file not found in the job directory.')
        self.flog = flog
        self.log = None
        
        # STRU.cif
        fn_stru_cif = os.path.join(self.outdir, 'STRU.cif')
        if not os.path.exists(fn_stru_cif):
            raise FileNotFoundError('`STRU.cif` file not found in the job directory.')
        self.fn_stru_cif = fn_stru_cif
        self.stru_cif = None
        
        # istate.info
        fn_istate = os.path.join(self.outdir, 'istate.info')
        if not os.path.exists(fn_istate):
            raise FileNotFoundError('`istate.info` file not found in the job directory.')
        self.fn_istate = fn_istate
        self.istate = None

        # STRU_ION_D
        fn_stru_ion_d = os.path.join(self.outdir, 'STRU_ION_D')
        if self.input.get('calculation', 'scf') not in ['scf', 'nscf'] and\
            not os.path.exists(fn_stru_ion_d):
            raise FileNotFoundError('`STRU_ION_D` file not found in the job directory.')
        self.fn_stru_ion_d = fn_stru_ion_d if os.path.exists(fn_stru_ion_d) else None
        self.stru_ion_d = None

    def __init__(self, jobdir):
        '''initialize the AbacusJob instance'''
        self.stru = None
        self.cell = None
        self.atomic_symbols = None
        self.atomic_positions = None
        
        try:
            self.init_as_unfinished(jobdir)
        except FileNotFoundError:
            raise RuntimeError(f'{jobdir} is not a valid ABACUS job directory.')
        
        try:
            self.init_as_finished(jobdir)
            self.complete = True
        except FileNotFoundError:
            self.complete = False
            
    def read_stru(self, cache=True):
        '''read the structure from the jobdir'''
        if self.stru is None:
            stru = read_stru_(self.fn_stru)
            if cache:
                self.stru = stru
        return self.stru

    def read_kpt(self, cache=True):
        '''read the kpt from the jobdir'''
        if self.kpt is None:
            kpt = read_kpt(self.fn_kpt)
            if cache:
                self.kpt = kpt
        return self.kpt

    def get_cell(self, unit='angstrom', reload_=False):
        '''get the cell from the jobdir'''
        if self.cell is not None and not reload_:
            return self.cell

        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                               'be called with `cache=True` before calling '
                               'this function.')
        
        self.cell = np.array(self.stru['lat']['vec']) * \
                convert_length_unit(self.stru['lat']['const'], 
                            unit_from='bohr', 
                            unit_to=unit)
        
        return self.cell

    @staticmethod
    def __extractstrutauc__(stru, unit='angstrom'):
        '''extract the atomic positions from the structure dict'''
        pos = np.array([atom['coord'] 
                        for s in stru['species'] 
                        for atom in s['atom']])
        factor = 1 if stru['coord_type'].startswith('Cartesian') \
            else stru['lat']['const']
        factor *= 1 if 'angstrom' in stru['coord_type'] \
            else convert_length_unit(1, 'bohr', 'angstrom')
        pos = pos * factor
        return pos.reshape(-1, 3) * convert_length_unit(1, 'angstrom', unit)

    def get_atomic_positions(self, unit='angstrom', reload_=False):
        '''get the atomic positions from the jobdir'''
        if self.atomic_positions is not None and not reload_:
            return self.atomic_positions
        
        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                               'be called with `cache=True` before calling '
                               'this function.')

        self.atomic_positions = self.__extractstrutauc__(self.stru, unit)
        return self.atomic_positions

    def get_trajectories(self, unit='angstrom'):
        '''
        get the trajectories from the outdir
        '''
        task = self.input.get('calculation', 'scf')
        if task not in ['md', 'relax']:
            raise RuntimeError('the job is not a MD or relax job, '
                               'cannot get the trajectories.')
        if task == 'md':
            return [self.__extractstrutauc__(
                stru=read_stru_(os.path.join(self.outdir, 'STRU', f)),
                unit=unit)
            for f in os.listdir(os.path.join(self.outdir, 'STRU'))
            if re.match(r'STRU_MD_\d+', f)]
        elif task == 'relax':
            return [self.__extractstrutauc__(
                stru=read_stru_(os.path.join(self.outdir, f)),
                unit=unit)
            for f in os.listdir(self.outdir)
            if re.match(r'STRU_ION\d+_D', f)]
        else:
            raise RuntimeError('AbacusJob.get_trajectories: reaches an unreachable '
                'code line, please report this issue to the developer.')

    def get_atomic_symbols(self, reload_=False):
        '''get the atomic symbols from the jobdir'''
        if self.atomic_symbols is not None and not reload_:
            return self.atomic_symbols
        
        if self.stru is None:
            raise RuntimeError('the function `read_stru` should '
                               'be called with `cache=True` before calling '
                               'this function.')
        
        elem = [[s['symbol']] * s['natom'] for s in self.stru['species']]
        elem = [item for sublist in elem for item in sublist]
        self.atomic_symbols = elem
        
        return self.atomic_symbols

    def to_deepmd(self, path_=None, fmt='deepmd', mode='initial'):
        '''convert the jobdir to a DeePMD-kit format
        
        Parameters
        ----------
        path_ : str
            the path to save the converted files, if None, 
            the files will be saved in the current directory
        fmt : str
            the format to save the converted files, default is 'deepmd'
            which means the files will be saved in the DeePMD-kit format.
            All possible formats are listed in the DPData documentation.
        mode : str
            the mode of the job, can be 'initial', 'md' or 'relax'.
            The default is 'initial', which means the job is a SCF job.
            If the mode is 'md', the job is a Molecular Dynamics trajectory,
            and if the mode is 'relax', the job is a relaxation job.
        
        Returns
        -------
        dpdata.System or dpdata.LabeledSystem
        the converted DeePMD-kit format data, which can be used
        to train a DeePMD-kit model.
        '''
        try:
            from dpdata import System, LabeledSystem
        except ImportError:
            raise ImportError('DPData is not installed. '
                              'Please install it with `pip install dpdata`.')
        
        assert re.match(r'deepmd(/\w+)?', fmt)
        assert mode in ['initial', 'md', 'relax']
        
        if mode == 'initial':
            mysys = System(file_name=self.fn_stru, 
                           fmt='abacus/stru')
        elif mode == 'md':
            mysys = LabeledSystem(file_name=self.path, 
                                  fmt='abacus/md')
        elif mode == 'relax':
            raise NotImplementedError('the mode `relax` is not well-implemented'
                                      'yet.')
            mysys = LabeledSystem(file_name=self.path, 
                                  fmt='abacus/relax')
        else:
            raise RuntimeError('AbacusJob.to_deepmd: reaches an unreachable '
                'code line, please report this issue to the developer.')
        if path_:
            mysys.to(fmt=fmt, file_name=path_)
        return mysys

    def run(self, 
            command, 
            remote=None,
            walltime=None,
            reload_after_run=True):
        '''
        run the ABACUS job
        
        Parameters
        ----------
        command : str
            the command to run the ABACUS job, such as 
            `OMP_NUM_THREADS=1 mpirun -np 16 abacus`. The stdout
            and stderr will be written to `out.log` file in the job directory.
        remote : dict
            the settings to run the ABACUS job on a remote server
        '''
        if not remote:
            cwd = os.getcwd() # backup the current directory
            os.chdir(self.path)
            # open a subprocess to run the command
            with open('out.log', 'w') as f:
                process = subprocess.Popen(command, 
                                           shell=True, 
                                           stdout=f, 
                                           stderr=subprocess.STDOUT)
                if walltime:
                    process.wait(timeout=walltime)
                else:
                    process.wait()
            if process.returncode != 0:
                raise RuntimeError('ABACUS job failed with return code '
                    f'{process.returncode}. Please check the `out.log` '
                    'file for more details.')
            else:
                if reload_after_run:
                    try:
                        self.init_as_finished(self.path)
                        self.complete = True
                    except FileNotFoundError:
                        raise RuntimeError('ABACUS job is detected as '
                            'finished by Python.subprocess module, but'
                            ' there are files missing. Please check '
                            'the status of job manually.')
            os.chdir(cwd) # return to the original directory
        else:
            raise NotImplementedError('remote run is not implemented yet.')
            assert isinstance(remote, dict)
            assert remote['mode'] == 'abacustest' # only support abacustest

    def read_exc_terms(self, unit='eV') -> list:
        '''read the energy terms from the log file
        
        Parameters
        ----------
        unit : str
            the unit of the energy, default is 'eV'
        
        Returns
        -------
        list of list of float
            the energy terms, indexed with [iframe][iterm] -> float
        '''
        if self.flog is None:
            raise RuntimeError('the function `init_as_finished` should '
                               'be called before calling this function.')
        raise NotImplementedError('the function `read_exc_terms` is not implemented '
            'yet due to the unfinished development at ABACUS side. One must first '
            'clarify the energy terms in the ABACUS log file.')
        # it is recommended to extend the ABACUS first on a forked repo
        return read_energy(self.flog, unit=unit)

class AbacusJobTest(unittest.TestCase):
    
    def setUp(self):
        testfiles = os.path.dirname(__file__)
        testfiles = os.path.dirname(testfiles)
        self.testfiles = os.path.abspath(os.path.join(testfiles, 'testfiles'))
        self.scheduled_delete = [] # to remove in tearDown
    
    def tearDown(self):
        '''remove the temporary files and folders created during the test'''
        for f in self.scheduled_delete:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)
        self.scheduled_delete = []
    
    def test_init_as_unfinished(self):
        jobdir = os.path.join(self.testfiles, 'scf-unfinished')
        job = AbacusJob(jobdir)
        self.assertEqual(job.fn_input, os.path.join(jobdir, 'INPUT'))
        self.assertEqual(job.fn_stru, os.path.join(jobdir, 'STRU'))
        self.assertEqual(job.fn_kpt, os.path.join(jobdir, 'KPT'))
        self.assertEqual(job.path, jobdir)
        self.assertFalse(job.complete)
        
    def test_init_as_finished(self):
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        job = AbacusJob(jobdir)
        self.assertEqual(job.outdir, os.path.join(jobdir, 'OUT.ABACUS'))
        self.assertEqual(job.flog, os.path.join(job.outdir, 'running_scf.log'))
        self.assertEqual(job.fn_stru_cif, os.path.join(job.outdir, 'STRU.cif'))
        self.assertEqual(job.fn_istate, os.path.join(job.outdir, 'istate.info'))
        self.assertTrue(job.complete)
    
    def test_get_md_trajectories(self):
        jobdir = os.path.join(self.testfiles, 'md-finished')
        job = AbacusJob(jobdir)
        
        trajs = job.get_trajectories(unit='angstrom')
        self.assertIsInstance(trajs, list)
        self.assertEqual(len(trajs), 5)  # there are 5 MD steps
        for traj in trajs:
            self.assertIsInstance(traj, np.ndarray)
            self.assertEqual(traj.shape, (2, 3))  # 2 atoms, 3 coordinates
            
    def test_get_relax_trajectories(self):
        jobdir = os.path.join(self.testfiles, 'relax-finished')
        job = AbacusJob(jobdir)
        
        trajs = job.get_trajectories(unit='angstrom')
        self.assertIsInstance(trajs, list)
        self.assertEqual(len(trajs), 8)
        for traj in trajs:
            self.assertIsInstance(traj, np.ndarray)
            self.assertEqual(traj.shape, (2, 3))
    
    def test_to_deepmd_scf(self):
        try:
            import dpdata
        except ImportError:
            # skip the test if dpdata is not installed
            self.skipTest('DPData is not installed. The unittest of function '
                          '`to_deepmd` is skipped. '
                          'Please install it with `pip install dpdata`.')
        jobdir = os.path.join(self.testfiles, 'scf-finished')
        job = AbacusJob(jobdir)
        
        dppath = str(uuid.uuid4())
        os.makedirs(dppath, exist_ok=True)
        self.scheduled_delete.append(dppath)  # to remove in tearDown
        
        out = job.to_deepmd(dppath)
        self.assertIsInstance(out, dpdata.System)
        self.assertTrue(os.path.exists(dppath))
        self.assertTrue(all(os.path.exists(os.path.join(dppath, f))
                        for f in ['box.raw', 'coord.raw', 'type_map.raw', 'type.raw']))
        
        # test the case to_deepmd with fmt='deepmd/npy'
        dppath_npy = str(uuid.uuid4())
        os.makedirs(dppath_npy, exist_ok=True)
        self.scheduled_delete.append(dppath_npy)  # to remove in tearDown
        
        out = job.to_deepmd(path_=dppath_npy, fmt='deepmd/npy')
        self.assertIsInstance(out, dpdata.System)
        self.assertTrue(os.path.exists(dppath_npy))
        self.assertTrue(all(os.path.exists(os.path.join(dppath_npy, f))
                        for f in ['type_map.raw', 'type.raw']))
        self.assertTrue(all(os.path.exists(os.path.join(dppath_npy, 'set.000', f))
                        for f in ['box.npy', 'coord.npy', 'move.npy']))
        # because the set.*** will split the data into several folders, 
        # usually we do not consider this format

    def test_to_deepmd_md(self):
        '''test the case that convert the Molecular Dynamics trajectory
        to the DeePMD-kit compatiable format'''
        try:
            import dpdata
        except ImportError:
            # skip the test if dpdata is not installed
            self.skipTest('DPData is not installed. The unittest of function '
                          '`to_deepmd` is skipped. '
                          'Please install it with `pip install dpdata`.')
        jobdir = os.path.join(self.testfiles, 'md-finished')
        job = AbacusJob(jobdir)
        
        dppath = str(uuid.uuid4())
        os.makedirs(dppath, exist_ok=True)
        self.scheduled_delete.append(dppath)  # to remove in tearDown
        
        out = job.to_deepmd(mode='md', path_=dppath)
        self.assertIsInstance(out, dpdata.LabeledSystem)
        self.assertTrue(os.path.exists(dppath))
        self.assertTrue(all(os.path.exists(os.path.join(dppath, f))
                        for f in ['box.raw', 'coord.raw', 'type_map.raw', 'type.raw']))

    def test_build_derived(self):
        jobdir = os.path.join(self.testfiles, 'scf-unfinished')
        job = AbacusJob(jobdir)
        _ = job.read_stru(cache=True)
        _ = job.read_kpt(cache=True)
        
        jobdir_derived = 'AbacusJobDerivedTest'
        self.scheduled_delete.append(jobdir_derived)  # to remove in tearDown
        
        dftparam = {'calculation': 'nscf', 'kspacing': 0.2}
        
        jobnew = job.build_derived(jobdir=jobdir_derived,
                                   dftparam=dftparam,
                                   instatiate=True)
        self.assertTrue(os.path.exists(jobdir_derived))
        self.assertIsInstance(jobnew, AbacusJob)
        self.assertEqual(jobnew.path, jobdir_derived)
        self.assertEqual(jobnew.input['calculation'], 'nscf')
        self.assertEqual(jobnew.input['kspacing'], '0.2')
        
    @unittest.skip('Example/Integrated test: this test requires a real ABACUS job to run, '
                   'it will be quite time-consuming, so skip it by default.')
    def test_run(self):
        jobdir = os.path.join(self.testfiles, 'scf-unfinished')
        job = AbacusJob(jobdir)

        # run the job
        command = 'OMP_NUM_THREADS=1 mpirun -np 1 abacus'
        with self.assertRaises(RuntimeError) as cm:
            job.run(command)
        # because the pseudopotential and orbital files are not provided
        
        # check the output then remove
        self.assertTrue(os.path.exists(os.path.join(job.path, 'out.log')))
        os.remove(os.path.join(job.path, 'out.log'))
        self.assertTrue(os.path.exists(os.path.join(job.path, 'time.json')))
        os.remove(os.path.join(job.path, 'time.json'))
        self.assertTrue(os.path.exists(os.path.join(job.path, 'OUT.ABACUS')))
        shutil.rmtree(os.path.join(job.path, 'OUT.ABACUS'))

if __name__ == '__main__':
    unittest.main()
