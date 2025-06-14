#!/usr/bin/bash

set -e # Exit on error

# abacus interface
python3 ./UniMolXC/abacus/control.py -v
python3 ./UniMolXC/abacus/inputio.py -v
python3 ./UniMolXC/abacus/struio.py -v

# geometry
python3 ./UniMolXC/geometry/manip/cluster.py -v
python3 ./UniMolXC/geometry/repr/_unimol.py -v
python3 ./UniMolXC/geometry/repr/_deepmd.py -v

# network
python3 ./UniMolXC/network/utility/xcfit.py -v
python3 ./UniMolXC/network/utility/xcloss.py -v
python3 ./UniMolXC/network/kernel/_unimol.py -v
python3 ./UniMolXC/network/kernel/_xcnet.py -v
