'''
read and write the Gaussian cube file format.
'''

def read(fcube: str):
    """read the Gaussian cube format volumetric data.
    The detailed format information can be found here:
    https://paulbourke.net/dataformats/cube/

    In brief, 

    ```
    [comment line1]
    [comment line2]
    [natom] [x-origin] [y-origin] [z-origin]
    [nx] [e11 in a.u.] [e12 in a.u.] [e13 in a.u.]
    [ny] [e21 in a.u.] [e22 in a.u.] [e23 in a.u.]
    [nz] [e31 in a.u.] [e32 in a.u.] [e33 in a.u.]
    [Z1] [chg1] [x1 in a.u.] [y1 in a.u.] [z1 in a.u.]
    [Z2] [chg2] [x2 in a.u.] [y2 in a.u.] [z2 in a.u.]
    ...
    [Znatom] [xnatom in a.u.] [ynatom in a.u.] [znatom in a.u.]
    [data1] [data2] [data3] ... [data6]
    [data7] [data8] [data9] ... [data12]
    ...
    ```
    
    Args:
        fcube (str): the file name of the cube file.
    
    Returns:
        data (dict): a dictionary containing the following keys:
        - "comment 1": the first comment line
        - "comment 2": the second comment line
        - "natom": the number of atoms
        - "origin": a tuple of (x_origin, y_origin, z_origin)
        - "nx": the number of grid points in x direction
        - "ny": the number of grid points in y direction
        - "nz": the number of grid points in z direction
        - "R": a 3x3 numpy array representing the cell in Bohr
        - "atomz": a list of atomic numbers
        - "chg": a list of atomic charges
        - "coords": a list of coordinates, each coordinate is a list of [x, y, z]
        - "data": a numpy array of volumetric data, flattened (not reshaped)
    """
    import re
    import numpy as np
    with open(fcube, 'r') as f:
        lines = f.readlines()
    
    out = {}
    out["comment 1"] = lines[0].strip()
    out["comment 2"] = lines[1].strip()

    natom, x_origin, y_origin, z_origin = map(float, lines[2].split())
    out["natom"] = natom
    out["origin"] = (x_origin, y_origin, z_origin)

    nx, e11, e12, e13 = map(float, lines[3].split())
    ny, e21, e22, e23 = map(float, lines[4].split())
    nz, e31, e32, e33 = map(float, lines[5].split())
    out["nx"] = nx
    out["ny"] = ny
    out["nz"] = nz
    out["R"] = np.array([[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]])

    atomz, chg, coords = [], [], []
    for line in lines[6:6+int(natom)]:
        atomz.append(int(line.split()[0]))
        chg.append(float(line.split()[1]))
        coords.append(list(map(float, line.split()[2:])))
    out["atomz"] = atomz
    out["chg"] = chg
    out["coords"] = coords

    data = []
    for line in lines[6+int(natom):]:
        data.extend(list(map(float, line.split())))
    #out["data"] = np.array(data).reshape(int(nx), int(ny), int(nz)) # is it necessary to reshape?
    out["data"] = np.array(data) # not reshaped
    return out

def write(data: dict, fcube: str, **kwargs):
    """write the Gaussian cube format volumetric data.
    Data should be organized as what is returned by read_cube().
    
    (if present) the ndigits will be used to control the number of digits of volumetric data,
    while the coordination data in header of cube is fixed to 6 digits.
    
    format of each part is defined in the following with format string:
    ```
    %s
     %s
     %4d %11.6f %11.6f %11.6f
     %4d %11.6f %11.6f %11.6f
     %4d %11.6f %11.6f %11.6f
     %4d %11.6f %11.6f %11.6f
     %4d %11.6f %11.6f %11.6f %11.6f
    ...
     %[width].[ndigits]f %[width].[ndigits]f %[width].[ndigits]f %[width].[ndigits]f %[width].[ndigits]f %[width].[ndigits]f
    ...
    ```

    Args:
        data (dict): the dictionary containing the cube data.
        fcube (str): the file name of the cube file.
        **kwargs: other optional arguments.
    
    Returns:
        None
    """

    ndigits = kwargs.get("ndigits", 6)
    width = ndigits + 7
    # temporarily there is no more format controlling options.

    with open(fcube, 'w') as f:
        f.write(data["comment 1"] + "\n")
        f.write(data["comment 2"] + "\n")
        f.write(" %4d %11.6f %11.6f %11.6f\n" % (data["natom"], data["origin"][0], data["origin"][1], data["origin"][2]))
        f.write(" %4d %11.6f %11.6f %11.6f\n" % (data["nx"], data["R"][0][0], data["R"][0][1], data["R"][0][2]))
        f.write(" %4d %11.6f %11.6f %11.6f\n" % (data["ny"], data["R"][1][0], data["R"][1][1], data["R"][1][2]))
        f.write(" %4d %11.6f %11.6f %11.6f\n" % (data["nz"], data["R"][2][0], data["R"][2][1], data["R"][2][2]))
        for i in range(int(data["natom"])):
            f.write(" %4d %11.6f %11.6f %11.6f %11.6f\n" % (data["atomz"][i], data["chg"][i], data["coords"][i][0], data["coords"][i][1], data["coords"][i][2]))
        for i in range(0, len(data["data"]), 6):
            f.write(" ".join([f"%{width}.{ndigits}e" % x for x in data["data"][i:i+6]]))
            f.write("\n")
    return

