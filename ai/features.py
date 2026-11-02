"""Landmark feature layout shared by data collection, training and real-time inference.

One *frame vector* (FRAME_DIM floats) describes a single video frame:

    pose (9 upper-body points, xyz)           27   body-relative
    left hand  (21 points, xyz)               63   body-relative  -> where the hand is
    right hand (21 points, xyz)               63   body-relative
    left hand shape  (wrist-relative, xyz)    63   -> what the hand shape is
    right hand shape (wrist-relative, xyz)    63
    face (23 points, xy)                      46   facial expression / head pose
    presence flags (pose, left, right, face)   4

"Body-relative" means: origin = shoulder centre, unit = shoulder width, so the
features are independent of where the signer stands and how far from the camera.
A *sequence* is a (T, FRAME_DIM) array; `sequence_features` resamples it to a
fixed SEQ_LEN and appends frame-to-frame velocities (movement over time).
"""
from __future__ import annotations

import numpy as np

SEQ_LEN = 32

# --- MediaPipe landmark selections -------------------------------------------------------------
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]      # nose, shoulders, elbows, wrists, hips
POSE_NAMES = ["nose", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr", "l_hip", "r_hip"]
FACE_IDX = [1, 152, 13, 14, 10,                    # nose tip, chin, lips, forehead (centre line)
            33, 263, 133, 362, 61, 291,            # eye corners, mouth corners
            70, 300, 105, 334, 107, 336,           # eyebrows
            234, 454,                              # cheeks
            159, 386, 145, 374]                    # eyelids
N_POSE, N_FACE, N_HAND = len(POSE_IDX), len(FACE_IDX), 21

# --- block layout of one frame vector ----------------------------------------------------------
_sizes = {"pose": N_POSE * 3, "lh": N_HAND * 3, "rh": N_HAND * 3, "lh_shape": N_HAND * 3,
          "rh_shape": N_HAND * 3, "face": N_FACE * 2, "flags": 4}
SLICES: dict[str, slice] = {}
_o = 0
for _k, _n in _sizes.items():
    SLICES[_k] = slice(_o, _o + _n)
    _o += _n
FRAME_DIM = _o
BODY_DIMS = _sizes["pose"] + _sizes["lh"] + _sizes["rh"] + _sizes["face"]      # blocks that get velocities
FEAT_DIM = FRAME_DIM + BODY_DIMS

F_POSE, F_LH, F_RH, F_FACE = (FRAME_DIM - 4 + i for i in range(4))   # flag column indices

SHOULDER_L, SHOULDER_R = POSE_IDX.index(11), POSE_IDX.index(12)
HIP_L, HIP_R = POSE_IDX.index(23), POSE_IDX.index(24)


def _mirror_perm(names_a: list[int], pairs: list[tuple[int, int]]) -> np.ndarray:
    pos = {v: i for i, v in enumerate(names_a)}
    perm = np.arange(len(names_a))
    for a, b in pairs:
        perm[pos[a]], perm[pos[b]] = pos[b], pos[a]
    return perm


_POSE_PERM = _mirror_perm(POSE_IDX, [(11, 12), (13, 14), (15, 16), (23, 24)])
_FACE_PERM = _mirror_perm(FACE_IDX, [(33, 263), (133, 362), (61, 291), (70, 300), (105, 334), (107, 336),
                                     (234, 454), (159, 386), (145, 374)])


# --------------------------------------------------------------------------- #
# Raw MediaPipe arrays -> frame vector
# --------------------------------------------------------------------------- #
def frame_vector(pose: np.ndarray | None, left: np.ndarray | None, right: np.ndarray | None,
                 face: np.ndarray | None) -> np.ndarray:
    """Build one normalised frame vector.

    pose: (33,3) | None, left/right: (21,3) | None, face: (478,3) | None.
    All coordinates are MediaPipe image-normalised x,y(,z).
    """
    out = np.zeros(FRAME_DIM, dtype=np.float32)
    if pose is None or len(pose) < 25:
        return out                                    # without a body reference nothing can be normalised
    p = np.asarray(pose, dtype=np.float32)[POSE_IDX]
    centre = (p[SHOULDER_L] + p[SHOULDER_R]) / 2.0
    scale = float(np.linalg.norm((p[SHOULDER_L] - p[SHOULDER_R])[:2]))
    if scale < 1e-4:
        return out
    out[SLICES["pose"]] = ((p - centre) / scale).reshape(-1)
    out[F_POSE] = 1.0

    for hand, key, shape_key, flag in ((left, "lh", "lh_shape", F_LH), (right, "rh", "rh_shape", F_RH)):
        if hand is None:
            continue
        h = np.asarray(hand, dtype=np.float32)
        out[SLICES[key]] = ((h - centre) / scale).reshape(-1)
        rel = h - h[0]
        palm = float(np.linalg.norm((h[9] - h[0])[:2])) + 1e-6           # wrist -> middle-finger base
        out[SLICES[shape_key]] = (rel / palm).reshape(-1)
        out[flag] = 1.0

    if face is not None:
        f = np.asarray(face, dtype=np.float32)[FACE_IDX][:, :2]
        out[SLICES["face"]] = ((f - centre[:2]) / scale).reshape(-1)
        out[F_FACE] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #
def resample(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Linear time-resampling of a (T, D) sequence to (length, D)."""
    seq = np.asarray(seq, dtype=np.float32)
    t = len(seq)
    if t == 0:
        return np.zeros((length, seq.shape[1] if seq.ndim == 2 else FRAME_DIM), dtype=np.float32)
    if t == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((length, seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(dst, src, seq[:, c])
    out[:, -4:] = (out[:, -4:] > 0.5).astype(np.float32)               # flags stay binary
    return out


def sequence_features(seq: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """(T, FRAME_DIM) raw frame vectors -> (length, FEAT_DIM) model input (positions + velocities)."""
    s = resample(seq, length)
    vel = np.zeros((length, BODY_DIMS), dtype=np.float32)
    vel[1:] = s[1:, :BODY_DIMS] - s[:-1, :BODY_DIMS]
    # velocities are small numbers; scale up so they matter to the network
    return np.concatenate([s, vel * 5.0], axis=1).astype(np.float32)


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Mirror left<->right (data augmentation, lets left-handed signers be recognised)."""
    s = np.array(seq, dtype=np.float32, copy=True)
    t = s.shape[0]

    def xyz(block):            # view as (T, N, 3)
        return s[:, SLICES[block]].reshape(t, -1, 3)

    pose = xyz("pose")[:, _POSE_PERM].copy()
    pose[..., 0] *= -1
    s[:, SLICES["pose"]] = pose.reshape(t, -1)

    lh, rh = xyz("lh").copy(), xyz("rh").copy()
    lh[..., 0] *= -1
    rh[..., 0] *= -1
    s[:, SLICES["lh"]], s[:, SLICES["rh"]] = rh.reshape(t, -1), lh.reshape(t, -1)

    lhs, rhs = xyz("lh_shape").copy(), xyz("rh_shape").copy()
    lhs[..., 0] *= -1
    rhs[..., 0] *= -1
    s[:, SLICES["lh_shape"]], s[:, SLICES["rh_shape"]] = rhs.reshape(t, -1), lhs.reshape(t, -1)

    face = s[:, SLICES["face"]].reshape(t, -1, 2)[:, _FACE_PERM].copy()
    face[..., 0] *= -1
    s[:, SLICES["face"]] = face.reshape(t, -1)

    s[:, F_LH], s[:, F_RH] = s[:, F_RH].copy(), s[:, F_LH].copy()
    return s
