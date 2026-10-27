"""2D monocular toy for deformation-field variants of Deformable 3DGS (see thesis/SPEC.md, section 10).

Scene: 2D Shepp-Logan-like phantom made of small GT Gaussians, deformed by swirl + zoom + shift.
Observation: one 1D line-integral projection (128 px) per frame ("monocular"); camera either jumps to a
random angle every frame ("teleport", like D-NeRF) or rotates smoothly by 180 deg over the sequence.
Model: 300 2D Gaussians in canonical space + Deformable-3DGS-like MLP (PE without pi, skip connection,
heads dx, dtheta, dlogs zero-initialised, xyz and Sigma detached at the input), static warm-up.

usage: [ACT=relu|softplus|silu] [REPO_HEADS=1] python toy_dynamic_2d.py ENC COUPLED REG LAM CAMERA SEED
  ENC     : pe10 (repo PE, L=10) | pe4 | c2f10 (Nerfies coarse-to-fine) | ipe10 / ipe10k2 (integrated PE)
  COUPLED : 0 = shape from heads (baseline) | 1 = Sigma_t = F Sigma F^T, F from sigma points
  REG     : none | el (||F^T F - I||^2) | curv (second difference along the Gaussian's axes)
  LAM     : regulariser weight
  CAMERA  : teleport | smooth
  REPO_HEADS=1 : default-init shape heads and linear additive scale, as in gaussian_renderer
Prints one JSON line: train / test_fixed (6 fixed angles) / test_near (+-10, +-20 deg from the camera) PSNR,
and rough = median point-Jacobian norm / median secant norm at the Gaussian scale.
"""
import sys, json, time
import numpy as np
import jax, jax.numpy as jnp
import optax, os
REPO_HEADS = os.environ.get('REPO_HEADS') == '1'  # repo-like baseline: default-init shape heads + linear additive scale
ACT = os.environ.get('ACT', 'relu')  # relu | softplus | silu

variant, COUPLED, REG, LAM, camera, seed = sys.argv[1], sys.argv[2] == '1', sys.argv[3], float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
STEPS, WARM, BATCH, NG, PIX = 4000, 300, 8, 300, 128
T = 160
rng = np.random.default_rng(1234)  # scene is fixed; seed only changes model init / camera draws

# ---------------- ground truth ----------------
ellipses = [  # (cx, cy, ax, ay, angle, intensity)  Shepp-Logan-like
    (0.0, 0.0, 0.69, 0.92, 0.0, 1.0), (0.0, -0.0184, 0.6624, 0.874, 0.0, -0.8),
    (0.22, 0.0, 0.11, 0.31, -0.31, -0.2), (-0.22, 0.0, 0.16, 0.41, 0.31, -0.2),
    (0.0, 0.35, 0.21, 0.25, 0.0, 0.1), (0.0, 0.1, 0.046, 0.046, 0.0, 0.1),
    (-0.08, -0.605, 0.046, 0.023, 0.0, 0.1), (0.06, -0.605, 0.023, 0.046, 0.0, 0.1)]
g = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(g, g)
img = np.zeros_like(X)
for cx, cy, ax, ay, a, v in ellipses:
    xr = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
    yr = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
    img += v * ((xr / ax) ** 2 + (yr / ay) ** 2 <= 1)
mask = img > 1e-3
gt_m = np.stack([X[mask], Y[mask]], 1) * 0.8
gt_w = img[mask] * (g[1] - g[0]) ** 2 * 0.64
gt_s = (g[1] - g[0]) * 0.8 * 0.6


def phi_np(x, t):
    r2 = (x ** 2).sum(-1, keepdims=True)
    th = 0.6 * np.sin(2 * np.pi * t) * np.exp(-r2 / 0.3)
    c, s = np.cos(th), np.sin(th)
    y = np.concatenate([c * x[:, :1] - s * x[:, 1:], s * x[:, :1] + c * x[:, 1:]], 1)
    return y * (1 + 0.04 * np.sin(2 * np.pi * t)) + np.array([0.1 * np.sin(2 * np.pi * t), 0.05 * np.cos(2 * np.pi * t)])


det_s = np.linspace(-1.25, 1.25, PIX)
PIXVAR = (det_s[1] - det_s[0]) ** 2 / 4


def project_np(m, C, w, a):
    u = np.array([np.cos(a), np.sin(a)])
    mu1 = m @ u
    var = np.einsum('i,nij,j->n', u, C, u) + PIXVAR
    return (w[:, None] * np.exp(-(det_s[None] - mu1[:, None]) ** 2 / (2 * var[:, None])) / np.sqrt(2 * np.pi * var[:, None])).sum(0)


def gt_frame(t, a):
    m = phi_np(gt_m, t)
    e = 1e-4
    J = np.stack([(phi_np(gt_m + e * np.eye(2)[k], t) - phi_np(gt_m - e * np.eye(2)[k], t)) / (2 * e) for k in range(2)], 2)
    C = gt_s ** 2 * J @ J.transpose(0, 2, 1)
    return project_np(m, C, gt_w, a)


ts = np.arange(T) / T
train_idx, test_idx = np.arange(0, T, 2), np.arange(1, T, 2)
crng = np.random.default_rng(seed)
if camera == 'teleport':
    cam = crng.uniform(0, np.pi, T)
else:
    cam = np.pi * ts
train_t, train_a = ts[train_idx], cam[train_idx]
train_y = np.stack([gt_frame(t, a) for t, a in zip(train_t, train_a)])
fixed_angles = np.deg2rad([0, 30, 60, 90, 120, 150])
test_fixed = np.stack([[gt_frame(ts[i], a) for a in fixed_angles] for i in test_idx])          # (80,6,PIX)
near_offsets = np.deg2rad([-20, -10, 10, 20])
test_near = np.stack([[gt_frame(ts[i], cam[i] + d) for d in near_offsets] for i in test_idx])  # (80,4,PIX)
peak = max(train_y.max(), test_fixed.max())
train_y, test_fixed, test_near = train_y / peak, test_fixed / peak, test_near / peak
MASS = gt_w.sum() / peak
peak = 1.0

# ---------------- model ----------------
L = 4 if variant == 'pe4' else 10
LT = 6
W, D = 128, 4
key = jax.random.PRNGKey(seed)


def init_linear(k, fin, fout, zero=False):
    b = 1 / np.sqrt(fin)
    k1, k2 = jax.random.split(k)
    if zero:
        return (jnp.zeros((fin, fout)), jnp.zeros(fout))
    return (jax.random.uniform(k1, (fin, fout), minval=-b, maxval=b), jax.random.uniform(k2, (fout,), minval=-b, maxval=b))


xin, tin = 2 * (1 + 2 * L), 1 * (1 + 2 * LT)
ks = jax.random.split(key, 10)
skip = 2
ins = [xin + tin] + [W + xin + tin if i == skip else W for i in range(1, D)]
net = {'layers': [init_linear(ks[i], ins[i], W) for i in range(D)],
       'head': init_linear(ks[8], W, 5, zero=True)}   # dx(2), dtheta(1), dlogs(2); zero init for all variants
if REPO_HEADS:
    Wd, bd = init_linear(ks[7], W, 5)
    net['head'] = (net['head'][0].at[:, 2:].set(Wd[:, 2:]), net['head'][1].at[2:].set(bd[2:]))
k1, k2, k3 = jax.random.split(ks[9], 3)
r = 0.85 * jnp.sqrt(jax.random.uniform(k1, (NG,)))
ang = jax.random.uniform(k2, (NG,)) * 2 * jnp.pi
gs = {'mu': jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang)], 1),
      'logs': jnp.full((NG, 2), np.log(0.04)),
      'theta': jax.random.uniform(k3, (NG,)) * np.pi,
      'w': jnp.full((NG,), float(np.log(np.expm1(MASS / NG))))}
params = {'net': net, 'gs': gs}
freqs = 2.0 ** np.arange(L)
tfreqs = 2.0 ** np.arange(LT)


def enc_x(mu, var, alpha):
    # mu (N,2), var (N,2) diag of Sigma (detached) ; alpha = c2f window value
    xf = mu[:, :, None] * freqs                         # (N,2,L)
    if variant.startswith('ipe'):
        kap = 4.0 if variant == 'ipe10k2' else 1.0
        att = jnp.exp(-0.5 * kap * var[:, :, None] * freqs ** 2)
    else:
        att = jnp.ones_like(xf)
    if variant == 'c2f10':
        j = np.arange(L)
        att = att * (1 - jnp.cos(jnp.pi * jnp.clip(alpha - j, 0, 1))) / 2
    feats = jnp.concatenate([jnp.sin(xf) * att, jnp.cos(xf) * att], -1).reshape(mu.shape[0], -1)
    return jnp.concatenate([mu, feats], -1)


def enc_t(t):
    tf = t[:, None] * tfreqs
    return jnp.concatenate([t[:, None], jnp.sin(tf), jnp.cos(tf)], -1)


def mlp(net, xe, te):
    h = jnp.concatenate([xe, te], -1)
    for i, (Wm, b) in enumerate(net['layers']):
        if i == skip:
            h = jnp.concatenate([xe, te, h], -1)
        h = {'relu': jax.nn.relu, 'softplus': jax.nn.softplus, 'silu': jax.nn.silu}[ACT](h @ Wm + b)
    Wm, b = net['head']
    return h @ Wm + b


def gauss_cov(logs, theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)
    S2 = jnp.exp(2 * logs)
    return R @ (S2[:, :, None] * jnp.swapaxes(R, 1, 2))


def deformed(params, t, alpha, use_def):
    """returns deformed means, covariances, masses and the regulariser value at time t"""
    gs = params['gs']
    C0 = gauss_cov(gs['logs'], gs['theta'])
    var = jax.lax.stop_gradient(jnp.stack([C0[:, 0, 0], C0[:, 1, 1]], 1))
    te = enc_t(jnp.full((NG,), t))
    mu0 = jax.lax.stop_gradient(gs['mu'])
    need_sig = COUPLED or REG != 'none'
    if need_sig:
        h = 2 ** 0.5
        c, s_ = jnp.cos(gs['theta']), jnp.sin(gs['theta'])
        R = jax.lax.stop_gradient(jnp.stack([jnp.stack([c, -s_], -1), jnp.stack([s_, c], -1)], -2))
        stp = jax.lax.stop_gradient(h * jnp.exp(gs['logs']))
        A = R * stp[:, None, :]
        pts = jnp.concatenate([mu0[:, None], mu0[:, None] + jnp.swapaxes(A, 1, 2), mu0[:, None] - jnp.swapaxes(A, 1, 2)], 1)
    else:
        pts = mu0[:, None]
    xe = jax.vmap(lambda p: enc_x(p, var, alpha), in_axes=1, out_axes=1)(pts)
    out = jax.vmap(lambda e: mlp(params['net'], e, te), in_axes=1, out_axes=1)(xe) * use_def   # (N,P,5)
    reg = 0.0
    if need_sig:
        phi = pts + out[:, :, :2]
        Dm = jnp.swapaxes((phi[:, 1:3] - phi[:, 3:5]) / 2, 1, 2)
        F = Dm @ jnp.swapaxes(R / stp[:, None, :], 1, 2)
        if REG == 'el':
            reg = jnp.mean(jnp.sum((jnp.swapaxes(F, 1, 2) @ F - jnp.eye(2)) ** 2, (1, 2)))
        elif REG == 'curv':   # second central difference along the Gaussian's own axes = neglected UT term
            cc = phi[:, 1:3] + phi[:, 3:5] - 2 * phi[:, :1]
            reg = jnp.mean(jnp.sum(cc ** 2, -1) / stp ** 2)
    if COUPLED:
        C = F @ C0 @ jnp.swapaxes(F, 1, 2)
        m = gs['mu'] + out[:, 0, :2]
    else:
        o = out[:, 0]
        m = gs['mu'] + o[:, :2]
        if REPO_HEADS:   # s_t = s + ds in linear space (as gaussian_renderer: get_scaling + d_scaling)
            st = jnp.abs(jnp.exp(gs['logs']) + o[:, 3:5]) + 1e-6
            C = gauss_cov(jnp.log(st), gs['theta'] + o[:, 2])
        else:
            C = gauss_cov(gs['logs'] + o[:, 3:5], gs['theta'] + o[:, 2])
    return m, C, jax.nn.softplus(gs['w']), reg


det_j = jnp.asarray(det_s)


def project(m, C, w, a):
    u = jnp.array([jnp.cos(a), jnp.sin(a)])
    mu1 = m @ u
    var = jnp.einsum('i,nij,j->n', u, C, u) + PIXVAR
    return (w[:, None] * jnp.exp(-(det_j[None] - mu1[:, None]) ** 2 / (2 * var[:, None])) / jnp.sqrt(2 * jnp.pi * var[:, None])).sum(0)


def render(params, t, a, alpha, use_def):
    m, C, w, reg = deformed(params, t, alpha, use_def)
    return project(m, C, w, a), reg


def loss_fn(params, tb, ab, yb, alpha, use_def):
    pred, reg = jax.vmap(lambda t, a: render(params, t, a, alpha, use_def))(tb, ab)
    return jnp.mean((pred - yb) ** 2) + LAM * jnp.mean(reg) * use_def


sched = optax.exponential_decay(1e-3, STEPS, 0.01)
label = jax.tree_util.tree_map_with_path(lambda p, _: 'net' if p[0].key == 'net' else p[1].key, params)
opt = optax.multi_transform({'net': optax.adam(sched), 'mu': optax.adam(2e-3), 'logs': optax.adam(1e-2),
                             'theta': optax.adam(1e-2), 'w': optax.adam(2e-2)}, label)
state = opt.init(params)


@jax.jit
def step(params, state, tb, ab, yb, alpha, use_def):
    l, gr = jax.value_and_grad(loss_fn)(params, tb, ab, yb, alpha, use_def)
    up, state = opt.update(gr, state, params)
    return optax.apply_updates(params, up), state, l


brng = np.random.default_rng(seed + 99)
t0 = time.time()
for it in range(STEPS):
    idx = brng.integers(0, len(train_idx), BATCH)
    alpha = L * min(1.0, it / (0.6 * STEPS))
    params, state, l = step(params, state, jnp.asarray(train_t[idx]), jnp.asarray(train_a[idx]),
                            jnp.asarray(train_y[idx]), alpha, float(it >= WARM))


def psnr(p, y):
    return float(10 * np.log10(peak ** 2 / np.mean((np.asarray(p) - y) ** 2)))


alpha = float(L)
rend = jax.jit(jax.vmap(lambda t, a: render(params, t, a, alpha, 1.0)[0]))
tr = rend(jnp.asarray(train_t), jnp.asarray(train_a))
tt = np.repeat(ts[test_idx], len(fixed_angles)); aa = np.tile(fixed_angles, len(test_idx))
fx = np.asarray(rend(jnp.asarray(tt), jnp.asarray(aa))).reshape(test_fixed.shape)
tt2 = np.repeat(ts[test_idx], len(near_offsets)); aa2 = (cam[test_idx][:, None] + near_offsets[None]).ravel()
nr = np.asarray(rend(jnp.asarray(tt2), jnp.asarray(aa2))).reshape(test_near.shape)

# roughness of the learned field at Gaussian scale: point Jacobian norm vs secant over +-sigma along axes
def dx_of(x, t):
    gs = params['gs']
    C0 = gauss_cov(gs['logs'], gs['theta'])
    var = jnp.stack([C0[:, 0, 0], C0[:, 1, 1]], 1)
    return mlp(params['net'], enc_x(x, var, alpha), enc_t(jnp.full((NG,), t)))[:, :2]

tq = 0.37
Jp = jax.jacfwd(lambda x: dx_of(x, tq).sum(0))(params['gs']['mu'])           # (2, N, 2): sum over N ok since rows independent
Jp = jnp.transpose(Jp, (1, 0, 2))
sig = jnp.exp(params['gs']['logs']).max(1)[:, None, None]
E = jnp.eye(2)
Js = jnp.stack([(dx_of(params['gs']['mu'] + sig[:, 0] * E[k], tq) - dx_of(params['gs']['mu'] - sig[:, 0] * E[k], tq)) / (2 * sig[:, 0])
                for k in range(2)], 2)
rough = float(jnp.median(jnp.linalg.norm(Jp, axis=(1, 2))) / jnp.median(jnp.linalg.norm(Js, axis=(1, 2))))

res = dict(variant=('' if ACT == 'relu' else ACT + '_') + ('repoheads_' if REPO_HEADS else '') + ('sig_' if COUPLED else '') + variant + ('' if REG == 'none' else f'_{REG}{LAM:g}'), camera=camera, seed=seed, train=psnr(tr, train_y), test_fixed=psnr(fx, test_fixed),
           test_near=psnr(nr, test_near), rough=rough, secs=round(time.time() - t0, 1),
           size_med=float(np.median(np.exp(np.asarray(params['gs']['logs'])).max(1))))
print(json.dumps(res))
