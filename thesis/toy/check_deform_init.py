"""Numpy re-implementation of utils/time_utils.py::DeformNetwork (is_blender=True and False),
with PyTorch's default nn.Linear init (U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias).
Forward-mode Jacobian of d_xyz w.r.t. x is propagated exactly through embedding, ReLU and skips.

Measures, for the untrained network (state at iteration warm_up = 3000, when deformation is switched on):
  |d_xyz|, |d_scaling|, |d_rotation|; point F = I + J: sigma_max and det F < 0;
  sigma-point F (h = sqrt 3) for several Gaussian sizes.
usage: python check_deform_init.py   (numpy only, ~2 min on CPU)
"""
import numpy as np

N = 4096


def linear(rng, fin, fout):
    b = 1 / np.sqrt(fin)
    return rng.uniform(-b, b, (fout, fin)), rng.uniform(-b, b, fout)


def embed(x, multires):
    """x (N,d) -> (N, d*(1+2L)), and d emb / d x as (N, E, d)."""
    N, d = x.shape
    feats, jac = [x], [np.broadcast_to(np.eye(d), (N, d, d))]
    for f in 2.0 ** np.linspace(0, multires - 1, multires):
        feats += [np.sin(x * f), np.cos(x * f)]
        jac += [np.einsum('ni,ij->nij', np.cos(x * f) * f, np.eye(d)),
                np.einsum('ni,ij->nij', -np.sin(x * f) * f, np.eye(d))]
    return np.concatenate(feats, 1), np.concatenate(jac, 1)


class Net:
    def __init__(self, seed, is_blender, D=8, W=256, multires=10):
        rng = np.random.default_rng(seed)
        self.is_blender, self.tm = is_blender, (6 if is_blender else 10)
        xin, tin = 3 * (1 + 2 * multires), 1 * (1 + 2 * self.tm)
        self.multires, self.skips = multires, [D // 2]
        if is_blender:
            self.time = [linear(rng, tin, 256), linear(rng, 256, 30)]
            tin = 30
        ins = [xin + tin] + [W + xin + tin if i in self.skips else W for i in range(D - 1)]
        self.layers = [linear(rng, i, W) for i in ins]
        self.warp, self.rot, self.scale = linear(rng, W, 3), linear(rng, W, 4), linear(rng, W, 3)

    def __call__(self, x, t, jac=False):
        te, _ = embed(t, self.tm)
        if self.is_blender:
            (W1, b1), (W2, b2) = self.time
            te = np.maximum(te @ W1.T + b1, 0) @ W2.T + b2
        xe, dxe = embed(x, self.multires)
        h = np.concatenate([xe, te], 1)
        dh = np.concatenate([dxe, np.zeros((len(x), te.shape[1], 3))], 1) if jac else None
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            m = h > 0
            h = h * m
            if jac:
                dh = np.einsum('oi,nid->nod', W, dh) * m[:, :, None]
            if i in self.skips:
                h = np.concatenate([xe, te, h], 1)
                if jac:
                    dh = np.concatenate([dxe, np.zeros((len(x), te.shape[1], 3)), dh], 1)
        out = [h @ W.T + b for W, b in (self.warp, self.rot, self.scale)]
        J = np.einsum('oi,nid->nod', self.warp[0], dh) if jac else None
        return out, J


def rand_rot(rng):
    q = rng.normal(size=(N, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    r, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
                     2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                     2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], 1).reshape(N, 3, 3)


def sigma_F(net, x, t, R, S, h=3 ** 0.5):
    A = R * (h * S)[:, None, :]
    rows = np.transpose(A, (0, 2, 1))
    pts = np.concatenate([x[:, None], x[:, None] + rows, x[:, None] - rows], 1).reshape(-1, 3)
    d = net(pts, np.repeat(t, 7, 0))[0][0].reshape(N, 7, 3)
    phi = pts.reshape(N, 7, 3) + d
    D = np.transpose((phi[:, 1:4] - phi[:, 4:7]) / 2, (0, 2, 1))
    return D @ np.linalg.inv(A)


def rep(tag, F):
    s = np.linalg.svd(F, compute_uv=False)[:, 0]
    det = np.linalg.det(F)
    print(f"  {tag:26s} sigma_max median {np.median(s):6.2f}  p90 {np.quantile(s, .9):6.2f}  det<0 {100 * np.mean(det < 0):5.1f}%")


for is_blender in (True, False):
    print(f"\n=== is_blender={is_blender} ===")
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        net = Net(seed, is_blender)
        v = rng.normal(size=(N, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        x = v * (0.8 + 0.05 * rng.normal(size=(N, 1)))
        t = rng.uniform(size=(N, 1))
        (dx, dr, ds), J = net(x, t, jac=True)
        F = np.eye(3) + J
        print(f" seed {seed}: |d_xyz| median {np.median(np.linalg.norm(dx, axis=1)):.3f}, "
              f"|d_scaling| median {np.median(np.abs(ds)):.3f}, |d_rotation| median {np.median(np.linalg.norm(dr, axis=1)):.3f}")
        # finite-difference check of the analytic Jacobian; a small step avoids crossing ReLU kinks
        e = 1e-7
        fd = np.stack([(net(x[:64] + e * np.eye(3)[j], t[:64])[0][0] - net(x[:64] - e * np.eye(3)[j], t[:64])[0][0]) / (2 * e)
                       for j in range(3)], 2)
        rel = np.median(np.linalg.norm((fd - J[:64]).reshape(64, -1), axis=1) / np.linalg.norm(J[:64].reshape(64, -1), axis=1))
        rep(f"point F (fd chk {rel:.0e})", F)
        R = rand_rot(rng)
        for s in (0.003, 0.01, 0.03):
            S = np.maximum(np.full((N, 3), s) * np.array([1.0, 0.4, 0.1]), 1e-3 / 3 ** 0.5)
            rep(f"sigma F, s_max={s}", sigma_F(net, x, t, R, S))
