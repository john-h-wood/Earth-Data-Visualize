import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

X = np.linspace(-1, 1, 1_000)
Y = np.linspace(-1, 1, 1_000)
xx, yy = np.meshgrid(X, Y)
Z = np.sin(9.4 * (xx ** 2 + yy ** 2))

# Setup plot
plt.rcParams.update({'text.usetex': True, 'font.size': 14})
plt.figure(figsize=(8, 8), dpi=300)

# Plotting
ax = plt.gca()
cs = ax.contourf(X, Y, Z, levels=(0.5, 1))

# Buddy's code
paths = cs.collections[0].get_paths()
for p in paths:
    sign = 1
    verts = p.vertices
    codes = p.codes
    idx = np.where(codes == Path.MOVETO)[0]
    vert_segs = np.split(verts, idx)[1:]
    code_segs = np.split(codes, idx)[1:]
    for code, vert in zip(code_segs, vert_segs):

        ##visualising (not necessary for the calculation)
        new_path = Path(vert, code)
        patch = PathPatch(
            new_path,
            edgecolor='black' if sign == 1 else 'red',
            facecolor='none',
            lw=1
        )
        ax.add_patch(patch)
        sign = -1 ##<-- assures that the other (inner) polygons will be subtracted

plt.savefig('contour_ex.png')
# It works!!
