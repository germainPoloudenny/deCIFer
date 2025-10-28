import os, gzip, pickle
from glob import glob
import h5py
from tqdm import tqdm
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element

input_dir = "cod_output_080124"
output_path = "../crystallography/data/CHILI-100K.pkl.gz"
records = []

for h5_path in tqdm(sorted(glob(os.path.join(input_dir, "*.h5")))):
    name = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as f:
        a, b, c, alpha, beta, gamma = f["GlobalLabels/CellParameters"][:]
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        frac_coords = f["UnitCellGraph/FractionalCoordinates"][:]
        z_numbers = f["UnitCellGraph/NodeFeatures"][:, 0]
        species = [Element.from_Z(int(round(z))).symbol for z in z_numbers]

        structure = Structure(lattice, species, frac_coords, to_unit_cell=True)
        cif_string = str(structure.to(fmt="cif"))

    records.append((name, cif_string))

with gzip.open(output_path, "wb") as f:
    pickle.dump(records, f)
print(f"Wrote {len(records)} entries to {output_path}")