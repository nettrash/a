"""Writes scipy_rotation.txt: SciPy's Rotation and Slerp on inputs that `rotation.rs` and
`euler.rs` generate again from the same seeds (see gen_common.py), or that are written here.

Quaternions are written with the scalar first, [w, x, y, z], as the crate has them; SciPy's are
[x, y, z, w]. Run with SciPy installed: `python3 -B gen_scipy.py` in this directory.
"""

import math
import warnings

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R, Slerp

from gen_common import Rng, line

warnings.simplefilter("ignore")  # SciPy warns at gimbal lock, which is tested on purpose.

SEQS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]
FRAMES = ["intrinsic", "extrinsic"]


def seqstr(s, frame):
    return s if frame == "intrinsic" else s.lower()


def wxyz(r):
    x, y, z, w = r.as_quat()
    return [w, x, y, z]


def rot(q):
    w, x, y, z = q
    return R.from_quat([x, y, z, w])


out = ["# SciPy " + scipy.__version__ + " Rotation and Slerp; written by gen_scipy.py"]

# Euler angles to quaternions: three triples of angles in [-4, 4) for each convention.
rng = Rng(1)
for s in SEQS:
    for f in FRAMES:
        for _ in range(3):
            a = [4.0 * rng.signed() for _ in range(3)]
            out.append(line("euler_from", wxyz(R.from_euler(seqstr(s, f), a))))

# Next to SciPy's gimbal tolerance of 1e-7: 0.9e-7 and 1.1e-7 from each singular value.
rng = Rng(12)
for s, f in [("XYZ", "intrinsic"), ("ZXZ", "extrinsic")]:
    proper = s[0] == s[2]
    lo, hi = (0.0, math.pi) if proper else (-math.pi / 2, math.pi / 2)
    for mid in [lo + 0.9e-7, lo + 1.1e-7, hi - 0.9e-7, hi - 1.1e-7]:
        a = [math.pi * rng.signed(), mid, math.pi * rng.signed()]
        out.append(line("gimbal_edge", R.from_euler(seqstr(s, f), a).as_euler(seqstr(s, f))))

# Quaternions to Euler angles: three random quaternions, of parts in [-1, 1), for each.
rng = Rng(2)
for s in SEQS:
    for f in FRAMES:
        for _ in range(3):
            q = [rng.signed() for _ in range(4)]
            out.append(line("euler_to", rot(q).as_euler(seqstr(s, f))))

# Gimbal lock: the middle angle at each singular value, within 1e-9 of it (inside SciPy's
# tolerance of 1e-7), and 1e-6 from it (outside); the first and third angles in [-pi, pi).
rng = Rng(3)
for s in SEQS:
    proper = s[0] == s[2]
    lo, hi = (0.0, math.pi) if proper else (-math.pi / 2, math.pi / 2)
    for f in FRAMES:
        for mid in [lo, lo + 1e-9, hi, hi - 1e-9, lo + 1e-6, hi - 1e-6]:
            first = math.pi * rng.signed()
            third = math.pi * rng.signed()
            r = R.from_euler(seqstr(s, f), [first, mid, third])
            out.append(line("gimbal", r.as_euler(seqstr(s, f))))

# Rotation matrices of random quaternions, and SciPy's quaternion of each matrix.
rng = Rng(4)
for _ in range(12):
    q = [rng.signed() for _ in range(4)]
    m = rot(q).as_matrix()
    out.append(line("matrix", list(m.flatten()) + wxyz(R.from_matrix(m))))

# Matrices near the switches of Shepperd's method, and not quite orthogonal: each written whole.
specials = []
for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 1], [0, 1, 1e-8]]:
    u = np.array(axis, dtype=float) / np.linalg.norm(axis)
    specials.append(R.from_rotvec(math.pi * u).as_matrix())
    specials.append(R.from_rotvec((math.pi - 1e-7) * u).as_matrix())
specials.append(np.eye(3))
specials.append(R.from_rotvec([1e-9, -2e-9, 3e-9]).as_matrix())
rng = Rng(5)
for scale in [1e-9, 1e-6, 1e-3]:
    q = [rng.signed() for _ in range(4)]
    noise = np.array([[rng.signed() for _ in range(3)] for _ in range(3)])
    specials.append(rot(q).as_matrix() + scale * noise)
q = [rng.signed() for _ in range(4)]
specials.append(1.001 * rot(q).as_matrix())
for m in specials:
    out.append(line("from_matrix", list(m.flatten()) + wxyz(R.from_matrix(m))))

# Rotation vectors to quaternions, of lengths from 1e-300 to 10, and back.
rng = Rng(6)
scales = [1e-300, 1e-12, 1e-6, 1.0, 3.0, 10.0]
for c in range(18):
    v = [scales[c % 6] * rng.signed() for _ in range(3)]
    out.append(line("rotvec_from", wxyz(R.from_rotvec(v))))
for c in range(12):
    q = [rng.signed() for _ in range(4)]
    out.append(line("rotvec_to", rot(q).as_rotvec()))
for q in [[1.0, 1e-9, -2e-9, 0.0], [0.0, 1.0, 2.0, 3.0], [-1.0, 0.1, 0.2, 0.3], [1.0, 1e-300, 0.0, 0.0],
          [-2.0, 0.0, 0.0, 3e-12], [1e-12, -1.0, 0.5, 0.25]]:
    out.append(line("rotvec_special", q + list(rot(q).as_rotvec())))

# Rotating vectors, and by the inverse rotation.
rng = Rng(7)
for _ in range(12):
    q = [rng.signed() for _ in range(4)]
    v = [10.0 * rng.signed() for _ in range(3)]
    out.append(line("apply", list(rot(q).apply(v)) + list(rot(q).apply(v, inverse=True))))

# Slerp between random rotations at random times in [0, 1).
rng = Rng(8)
for _ in range(12):
    a = [rng.signed() for _ in range(4)]
    b = [rng.signed() for _ in range(4)]
    t = rng.uniform()
    s = Slerp([0.0, 1.0], R.concatenate([rot(a), rot(b)]))
    out.append(line("slerp", wxyz(s(t))))
# Nearly equal and nearly opposite quaternions, written whole, at t = 0, 1/4, 1/2 and 1.
rng = Rng(9)
for delta, flip in [(1e-9, 1.0), (1e-9, -1.0), (1e-4, -1.0), (3.0, 1.0)]:
    a = [rng.signed() for _ in range(4)]
    axis = np.array([rng.signed() for _ in range(3)])
    b = rot(a) * R.from_rotvec(delta * axis / np.linalg.norm(axis))
    b = [flip * t for t in wxyz(b)]
    for t in [0.0, 0.25, 0.5, 1.0]:
        s = Slerp([0.0, 1.0], R.concatenate([rot(a), rot(b)]))
        out.append(line("slerp_special", [t] + wxyz(rot(a)) + b + wxyz(s(t))))

# The angle of a rotation, and between two.
rng = Rng(10)
for _ in range(12):
    q = [rng.signed() for _ in range(4)]
    p = [rng.signed() for _ in range(4)]
    out.append(line("magnitude", [rot(q).magnitude(), (rot(q).inv() * rot(p)).magnitude()]))

# The shortest arc from one vector to another: align_vectors([a], [b]) takes b to a.
rng = Rng(11)
for _ in range(8):
    a = [rng.signed() for _ in range(3)]
    b = [rng.signed() for _ in range(3)]
    r, _ = R.align_vectors([a], [b])
    out.append(line("align", wxyz(r)))

with open("scipy_rotation.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
