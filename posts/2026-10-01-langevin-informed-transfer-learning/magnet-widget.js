// Magnetization lab: Langevin dynamics of a single-domain nanoparticle's magnetization m on the sphere,
//   dm = -grad_S U(m) dt + sqrt(2 kT) dW_S,   U(m) = -Ku m_z^2 + Kc (mx^2 my^2 + my^2 mz^2 + mz^2 mx^2),
// and LITL step 2 run in the browser: from n uniformly random directions and their energies U only
// (no trajectories, no forces), estimate the generator's eigenpairs with a fixed dictionary of 49
// polynomial features, then use them to forecast relaxation and to drive the swarm with the learned drift.
(function () {
  const root = document.getElementById("mag-lab");
  if (!root) return;

  // ------------------------------------------------------------------ features: monomials of degree 5 and 6
  const P = 6, EXPS = [];
  for (let a = 0; a <= P; a++) for (let b = 0; b <= P; b++) for (let c = 0; c <= P; c++)
    if (a + b + c === P || a + b + c === P - 1) EXPS.push([a, b, c]);
  const F = EXPS.length; // 49
  const px = new Float64Array(P + 1), py = new Float64Array(P + 1), pz = new Float64Array(P + 1);
  function powers(x, y, z) {
    px[0] = py[0] = pz[0] = 1;
    for (let k = 1; k <= P; k++) { px[k] = px[k - 1] * x; py[k] = py[k - 1] * y; pz[k] = pz[k - 1] * z; }
  }
  function feat(x, y, z, out) {
    powers(x, y, z);
    for (let j = 0; j < F; j++) { const e = EXPS[j]; out[j] = px[e[0]] * py[e[1]] * pz[e[2]]; }
  }
  function featGrad(x, y, z, f, gx, gy, gz) {   // features and tangential gradients
    powers(x, y, z);
    for (let j = 0; j < F; j++) {
      const a = EXPS[j][0], b = EXPS[j][1], c = EXPS[j][2];
      f[j] = px[a] * py[b] * pz[c];
      const dx = a ? a * px[a - 1] * py[b] * pz[c] : 0;
      const dy = b ? b * px[a] * py[b - 1] * pz[c] : 0;
      const dz = c ? c * px[a] * py[b] * pz[c - 1] : 0;
      const r = dx * x + dy * y + dz * z;
      gx[j] = dx - r * x; gy[j] = dy - r * y; gz[j] = dz - r * z;
    }
  }

  // ------------------------------------------------------------------ physics
  let Ku = 1.0, Kc = 0.3, kT = 0.3;
  const energy = (x, y, z) => -Ku * z * z + Kc * (x * x * y * y + y * y * z * z + z * z * x * x);
  function sgrad(x, y, z, g) {
    const gx = 2 * Kc * x * (y * y + z * z), gy = 2 * Kc * y * (x * x + z * z), gz = -2 * Ku * z + 2 * Kc * z * (x * x + y * y);
    const r = gx * x + gy * y + gz * z;
    g[0] = gx - r * x; g[1] = gy - r * y; g[2] = gz - r * z;
  }

  // ------------------------------------------------------------------ linear algebra (cyclic Jacobi)
  function jacobi(Ain, n) {
    const A = Float64Array.from(Ain), V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    for (let sweep = 0; sweep < 30; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
      if (off < 1e-26) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const th = (aqq - app) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
    const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = A[i * n + i];
    return { w, V };
  }

  // ------------------------------------------------------------------ LITL fit from samples (x_n, U(x_n))
  function fit(pts, n) {
    const f = new Float64Array(F), gx = new Float64Array(F), gy = new Float64Array(F), gz = new Float64Array(F);
    const C = new Float64Array(F * F), D = new Float64Array(F * F), B = new Float64Array(F * 3);
    let Umin = Infinity; const Uv = new Float64Array(n);
    for (let s = 0; s < n; s++) { Uv[s] = energy(pts[3 * s], pts[3 * s + 1], pts[3 * s + 2]); if (Uv[s] < Umin) Umin = Uv[s]; }
    let vs = 0;
    for (let s = 0; s < n; s++) {
      const x = pts[3 * s], y = pts[3 * s + 1], z = pts[3 * s + 2];
      const v = Math.exp(-(Uv[s] - Umin) / kT); vs += v;           // importance weight e^{-beta U}
      featGrad(x, y, z, f, gx, gy, gz);
      for (let i = 0; i < F; i++) {
        const vfi = v * f[i], vgx = v * gx[i], vgy = v * gy[i], vgz = v * gz[i];
        B[3 * i] += vfi * x; B[3 * i + 1] += vfi * y; B[3 * i + 2] += vfi * z;
        for (let j = i; j < F; j++) {
          C[i * F + j] += vfi * f[j];
          D[i * F + j] += vgx * gx[j] + vgy * gy[j] + vgz * gz[j];
        }
      }
    }
    for (let i = 0; i < F; i++) for (let j = i; j < F; j++) {
      C[i * F + j] /= vs; D[i * F + j] *= kT / vs;                    // Dirichlet form: (1/beta) E_pi[grad.grad]
      C[j * F + i] = C[i * F + j]; D[j * F + i] = D[i * F + j];
    }
    for (let k = 0; k < 3 * F; k++) B[k] /= vs;
    // whiten C, then symmetric eigenproblem
    const ec = jacobi(C, F); let smax = 0; for (let i = 0; i < F; i++) smax = Math.max(smax, ec.w[i]);
    const keep = []; for (let i = 0; i < F; i++) if (ec.w[i] > smax * 1e-11) keep.push(i);
    const r = keep.length, T = new Float64Array(F * r);
    keep.forEach((i, c) => { const sc = 1 / Math.sqrt(ec.w[i]); for (let k = 0; k < F; k++) T[k * r + c] = ec.V[k * F + i] * sc; });
    const DT = new Float64Array(F * r);
    for (let i = 0; i < F; i++) for (let c = 0; c < r; c++) { let s = 0; for (let k = 0; k < F; k++) s += D[i * F + k] * T[k * r + c]; DT[i * r + c] = s; }
    const K = new Float64Array(r * r);
    for (let a = 0; a < r; a++) for (let b = a; b < r; b++) { let s = 0; for (let k = 0; k < F; k++) s += T[k * r + a] * DT[k * r + b]; K[a * r + b] = K[b * r + a] = s; }
    const ek = jacobi(K, r);
    const order = Array.from({ length: r }, (_, i) => i).sort((a, b) => ek.w[a] - ek.w[b]);
    const lam = new Float64Array(r), coef = new Float64Array(F * r);  // coef[:, i] = i-th eigenfunction
    order.forEach((o, i) => {
      lam[i] = -Math.max(ek.w[o], 0);
      for (let k = 0; k < F; k++) { let s = 0; for (let c = 0; c < r; c++) s += T[k * r + c] * ek.V[c * r + o]; coef[k * r + i] = s; }
    });
    const Em = new Float64Array(r * 3);                                // E_pi[psi_i m]
    for (let i = 0; i < r; i++) for (let d = 0; d < 3; d++) { let s = 0; for (let k = 0; k < F; k++) s += coef[k * r + i] * B[3 * k + d]; Em[3 * i + d] = s; }
    return { lam, coef, r, Em };
  }

  // ------------------------------------------------------------------ grids for display and learned drift
  const NT = 91, NP = 180;
  const grid = { U: new Float64Array(NT * NP), psi: [], drift: new Float64Array(NT * NP * 3) };
  for (let k = 0; k <= 8; k++) grid.psi.push(new Float64Array(NT * NP));
  function fillGrids(model) {
    const f = new Float64Array(F), r = model.r, ps = new Float64Array(r);
    for (let it = 0; it < NT; it++) {
      const th = Math.PI * it / (NT - 1), st = Math.sin(th), ct = Math.cos(th);
      for (let ip = 0; ip < NP; ip++) {
        const ph = 2 * Math.PI * ip / NP, x = st * Math.cos(ph), y = st * Math.sin(ph), z = ct, g = it * NP + ip;
        grid.U[g] = energy(x, y, z);
        feat(x, y, z, f);
        for (let i = 0; i < r; i++) { let s = 0; for (let k = 0; k < F; k++) s += f[k] * model.coef[k * r + i]; ps[i] = s; }
        for (let k = 1; k <= 8; k++) grid.psi[k][g] = k < r ? ps[k] : 0;
        let Gx = 0, Gy = 0, Gz = 0;
        for (let i = 1; i < r; i++) { const a = -model.lam[i] * ps[i]; Gx += a * model.Em[3 * i]; Gy += a * model.Em[3 * i + 1]; Gz += a * model.Em[3 * i + 2]; }
        const rr = Gx * x + Gy * y + Gz * z;                            // learned grad_S U = -(I - mm^T) L^ m
        grid.drift[3 * g] = Gx - rr * x; grid.drift[3 * g + 1] = Gy - rr * y; grid.drift[3 * g + 2] = Gz - rr * z;
      }
    }
    // fix the arbitrary sign of each eigenfunction: positive at the north pole (or at its max |value|)
    for (let k = 1; k <= 8; k++) {
      const a = grid.psi[k]; let im = 0; for (let g = 0; g < a.length; g++) if (Math.abs(a[g]) > Math.abs(a[im])) im = g;
      const sgn = Math.abs(a[0]) > 0.2 * Math.abs(a[im]) ? Math.sign(a[0]) : Math.sign(a[im]);
      if (sgn < 0) for (let g = 0; g < a.length; g++) a[g] = -a[g];
    }
  }
  function lookup(arr, x, y, z, stride, comp) {
    const th = Math.acos(Math.max(-1, Math.min(1, z))) / Math.PI * (NT - 1);
    let ph = Math.atan2(y, x); if (ph < 0) ph += 2 * Math.PI; ph = ph / (2 * Math.PI) * NP;
    const i0 = Math.min(NT - 2, Math.floor(th)), j0 = Math.floor(ph) % NP, j1 = (j0 + 1) % NP;
    const ti = th - i0, tj = ph - Math.floor(ph);
    const v = (i, j) => arr[(i * NP + j) * stride + comp];
    return (1 - ti) * ((1 - tj) * v(i0, j0) + tj * v(i0, j1)) + ti * ((1 - tj) * v(i0 + 1, j0) + tj * v(i0 + 1, j1));
  }

  // ------------------------------------------------------------------ samples
  let seed = 7;
  function rng() { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
  let spare = null;
  function gauss() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0; while (u === 0) u = Math.random(); v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u)); spare = r * Math.sin(2 * Math.PI * v); return r * Math.cos(2 * Math.PI * v);
  }
  function uniformSamples(n) {
    const pts = new Float64Array(3 * n);
    for (let s = 0; s < n; s++) {
      const z = 2 * rng() - 1, ph = 2 * Math.PI * rng(), r = Math.sqrt(1 - z * z);
      pts[3 * s] = r * Math.cos(ph); pts[3 * s + 1] = r * Math.sin(ph); pts[3 * s + 2] = z;
    }
    return pts;
  }
  const NF = 6000, fibPts = new Float64Array(3 * NF);
  for (let i = 0; i < NF; i++) {
    const z = 1 - 2 * (i + 0.5) / NF, r = Math.sqrt(1 - z * z), ph = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
    fibPts[3 * i] = r * Math.cos(ph); fibPts[3 * i + 1] = r * Math.sin(ph); fibPts[3 * i + 2] = z;
  }

  // ------------------------------------------------------------------ state
  let nBudget = 2000, colorBy = 0, useLearned = false, paused = false;
  let model = null, ref = null;
  const NSW = 500, sw = new Float64Array(3 * NSW);
  let simT = 0, relax = [], relaxOn = false, Tmax = 10;

  // ------------------------------------------------------------------ DOM
  const q = (s) => root.querySelector(s);
  const base = q("#mag-base"), over = q("#mag-over"), spec = q("#mag-spec"), rel = q("#mag-relax"), out = q("#mag-out");
  const RES = 560; base.width = base.height = over.width = over.height = RES;
  const bctx = base.getContext("2d"), octx = over.getContext("2d");
  const img = bctx.createImageData(RES, RES);
  const C0 = { ink: "#1f2328", ink2: "#57606a", grid: "#e6e8eb", orange: "#eb6834", blue: "#2a78d6", gray: "#8c959f" };

  // view rotation (world -> view)
  let R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const mul = (A, B) => { const Cm = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let s = 0; for (let k = 0; k < 3; k++) s += A[3 * i + k] * B[3 * k + j]; Cm[3 * i + j] = s; } return Cm; };
  const rotX = (a) => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];
  const rotY = (a) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const rotZ = (a) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
  R = mul(rotX(-1.15), rotZ(0.5));

  // colour maps
  const hex = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  const SEQ = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f5f9fe"].map(hex);
  const NEG = hex("#2a78d6"), MID = hex("#f0efec"), POS = hex("#eb6834");
  function seqColor(t) { t = Math.max(0, Math.min(1, t)) * (SEQ.length - 1); const i = Math.min(SEQ.length - 2, Math.floor(t)), f = t - i; return [0, 1, 2].map((k) => SEQ[i][k] + f * (SEQ[i + 1][k] - SEQ[i][k])); }
  function divColor(t) { const e = t < 0 ? NEG : POS, a = Math.min(1, Math.abs(t)); return [0, 1, 2].map((k) => MID[k] + a * (e[k] - MID[k])); }

  function renderSphere() {
    const d = img.data, half = RES / 2, rad = RES / 2 - 6;
    const field = colorBy === 0 ? grid.U : grid.psi[colorBy];
    let lo = Infinity, hi = -Infinity, amax = 0;
    for (let g = 0; g < field.length; g++) { lo = Math.min(lo, field[g]); hi = Math.max(hi, field[g]); amax = Math.max(amax, Math.abs(field[g])); }
    for (let py_ = 0; py_ < RES; py_++) for (let px_ = 0; px_ < RES; px_++) {
      const u = (px_ + 0.5 - half) / rad, v = -(py_ + 0.5 - half) / rad, rr = u * u + v * v, o = 4 * (py_ * RES + px_);
      if (rr > 1) { d[o + 3] = 0; continue; }
      const w = Math.sqrt(1 - rr);
      const x = R[0] * u + R[3] * v + R[6] * w, y = R[1] * u + R[4] * v + R[7] * w, z = R[2] * u + R[5] * v + R[8] * w;
      const val = lookup(field, x, y, z, 1, 0);
      const c = colorBy === 0 ? seqColor((val - lo) / (hi - lo || 1)) : divColor(val / (amax || 1));
      const sh = 0.6 + 0.4 * w;
      d[o] = c[0] * sh; d[o + 1] = c[1] * sh; d[o + 2] = c[2] * sh; d[o + 3] = rr > 0.985 ? 255 * (1 - rr) / 0.015 : 255;
    }
    bctx.putImageData(img, 0, 0);
  }

  function drawSwarm() {
    const half = RES / 2, rad = RES / 2 - 6;
    octx.clearRect(0, 0, RES, RES);
    // easy axis
    const ez = [R[2], R[5], R[8]];
    octx.strokeStyle = "rgba(31,35,40,0.55)"; octx.lineWidth = 2; octx.setLineDash([6, 6]);
    octx.beginPath(); octx.moveTo(half - ez[0] * rad * 1.08, half + ez[1] * rad * 1.08); octx.lineTo(half + ez[0] * rad * 1.08, half - ez[1] * rad * 1.08); octx.stroke();
    octx.setLineDash([]);
    for (let s = 0; s < NSW; s++) {
      const x = sw[3 * s], y = sw[3 * s + 1], z = sw[3 * s + 2];
      const u = R[0] * x + R[1] * y + R[2] * z, v = R[3] * x + R[4] * y + R[5] * z, w = R[6] * x + R[7] * y + R[8] * z;
      if (w < 0) continue;
      octx.beginPath(); octx.arc(half + u * rad, half - v * rad, 5.5, 0, 2 * Math.PI);
      octx.fillStyle = "#fff"; octx.fill();
      octx.beginPath(); octx.arc(half + u * rad, half - v * rad, 3.6, 0, 2 * Math.PI);
      octx.fillStyle = C0.ink; octx.fill();
    }
  }

  // ------------------------------------------------------------------ dynamics
  const g3 = [0, 0, 0];
  function step(dt) {
    const sq = Math.sqrt(2 * kT * dt);
    for (let s = 0; s < NSW; s++) {
      let x = sw[3 * s], y = sw[3 * s + 1], z = sw[3 * s + 2];
      if (useLearned) { g3[0] = lookup(grid.drift, x, y, z, 3, 0); g3[1] = lookup(grid.drift, x, y, z, 3, 1); g3[2] = lookup(grid.drift, x, y, z, 3, 2); }
      else sgrad(x, y, z, g3);
      let nx = gauss(), ny = gauss(), nz = gauss(); const r = nx * x + ny * y + nz * z; nx -= r * x; ny -= r * y; nz -= r * z;
      x += -g3[0] * dt + sq * nx; y += -g3[1] * dt + sq * ny; z += -g3[2] * dt + sq * nz;
      const nrm = Math.hypot(x, y, z); sw[3 * s] = x / nrm; sw[3 * s + 1] = y / nrm; sw[3 * s + 2] = z / nrm;
    }
  }
  function saturate() {
    for (let s = 0; s < NSW; s++) { sw[3 * s] = 0; sw[3 * s + 1] = 0; sw[3 * s + 2] = 1; }
    simT = 0; relax = [[0, 1]]; relaxOn = true;
  }
  function meanMz() { let s = 0; for (let k = 0; k < NSW; k++) s += sw[3 * k + 2]; return s / NSW; }

  function forecast(m, t) {   // E[m_z(t) | m_0 = north] = sum_i e^{lam_i t} psi_i(north) E_pi[psi_i m_z]
    let s = 0; const r = m.r;
    for (let i = 0; i < r; i++) s += Math.exp(m.lam[i] * t) * m.north[i] * m.Em[3 * i + 2];
    return s;
  }
  function addNorth(m) {
    const f = new Float64Array(F); feat(0, 0, 1, f); m.north = new Float64Array(m.r);
    for (let i = 0; i < m.r; i++) { let s = 0; for (let k = 0; k < F; k++) s += f[k] * m.coef[k * m.r + i]; m.north[i] = s; }
  }

  // ------------------------------------------------------------------ charts
  function drawSpectrum() {
    const W = 260, H = 150, L = 36, Rm = 8, T = 24, B = 32, K = 8;
    spec.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const tau = (l) => 1 / Math.max(Math.abs(l), 1e-9);
    const vals = []; for (let i = 1; i <= K; i++) vals.push(tau(ref.lam[i]), tau(model.lam[i]));
    const lmin = Math.floor(Math.log10(Math.min(...vals)) - 0.1), lmax = Math.ceil(Math.log10(Math.max(...vals)) + 0.05);
    const X = (i) => L + (W - L - Rm) * (i - 0.5) / K, Y = (t) => T + (H - T - B) * (1 - (Math.log10(t) - lmin) / (lmax - lmin));
    let s = `<text x="4" y="14" font-size="13" fill="${C0.ink2}">relaxation times 1/|λₖ|</text>`;
    if (colorBy > 0) s += `<rect x="${X(colorBy) - 13}" y="${T - 4}" width="26" height="${H - T - B + 8}" rx="5" fill="#fdf1ea"/>`;
    for (let e = lmin; e <= lmax; e++) s += `<line x1="${L}" x2="${W - Rm}" y1="${Y(10 ** e)}" y2="${Y(10 ** e)}" stroke="${C0.grid}"/><text x="${L - 6}" y="${Y(10 ** e) + 4}" font-size="12" fill="${C0.ink2}" text-anchor="end">${10 ** e >= 1 ? 10 ** e : (10 ** e).toFixed(Math.max(0, -e))}</text>`;
    for (let i = 1; i <= K; i++) {
      s += `<circle cx="${X(i)}" cy="${Y(tau(ref.lam[i]))}" r="6.5" fill="none" stroke="${C0.gray}" stroke-width="1.6"/>`;
      s += `<circle cx="${X(i)}" cy="${Y(tau(model.lam[i]))}" r="4.2" fill="${C0.orange}" stroke="#fff" stroke-width="1.5"/>`;
      s += `<text x="${X(i)}" y="${H - B + 16}" font-size="12.5" fill="${C0.ink2}" text-anchor="middle">${i}</text>`;
    }
    s += `<text x="${(L + W) / 2}" y="${H - 1}" font-size="12" fill="${C0.ink2}" text-anchor="middle">mode k</text>`;
    spec.innerHTML = s;
  }
  const rctx = rel.getContext("2d");
  function drawRelax() {
    const dpr = window.devicePixelRatio || 1, W = 260, H = 150;
    if (rel.width !== W * dpr) { rel.width = W * dpr; rel.height = H * dpr; }
    rctx.setTransform(dpr, 0, 0, dpr, 0, 0); rctx.clearRect(0, 0, W, H);
    const L = 36, Rm = 8, T = 24, B = 32;
    const X = (t) => L + (W - L - Rm) * t / Tmax, Y = (v) => T + (H - T - B) * (1.05 - v) / 1.3;
    rctx.font = "13px system-ui, sans-serif"; rctx.fillStyle = C0.ink2; rctx.textAlign = "left"; rctx.fillText("⟨m_z⟩ after saturating ↑", 4, 14);
    rctx.font = "12px system-ui, sans-serif"; rctx.textAlign = "right";
    for (const v of [0, 0.5, 1]) { rctx.strokeStyle = C0.grid; rctx.lineWidth = 1; rctx.beginPath(); rctx.moveTo(L, Y(v)); rctx.lineTo(W - Rm, Y(v)); rctx.stroke(); rctx.fillText(v.toFixed(1), L - 6, Y(v) + 4); }
    rctx.textAlign = "center";
    const tick = Tmax > 40 ? 20 : Tmax > 16 ? 10 : Tmax > 6 ? 2 : 1;
    for (let t = 0; t <= Tmax + 1e-9; t += tick) rctx.fillText(String(Math.round(t)), X(t), H - B + 16);
    rctx.fillText("time", (L + W) / 2, H - 1);
    // tau_1 marker
    const t1 = 1 / Math.abs(model.lam[1]);
    if (t1 < Tmax) { rctx.strokeStyle = C0.gray; rctx.setLineDash([3, 3]); rctx.beginPath(); rctx.moveTo(X(t1), T); rctx.lineTo(X(t1), H - B); rctx.stroke(); rctx.setLineDash([]); rctx.fillStyle = C0.ink2; rctx.textAlign = "left"; rctx.fillText("τ₁", X(t1) + 4, T + 10); }
    // forecasts
    const curve = (m, col, lw, dash) => { rctx.strokeStyle = col; rctx.lineWidth = lw; rctx.setLineDash(dash); rctx.beginPath(); for (let k = 0; k <= 120; k++) { const t = Tmax * k / 120; const v = forecast(m, t); k ? rctx.lineTo(X(t), Y(v)) : rctx.moveTo(X(t), Y(v)); } rctx.stroke(); rctx.setLineDash([]); };
    curve(ref, C0.gray, 1.4, [4, 3]);
    curve(model, C0.orange, 2.2, []);
    rctx.fillStyle = C0.ink;
    for (const [t, v] of relax) { if (t > Tmax) break; rctx.beginPath(); rctx.arc(X(t), Y(v), 2.2, 0, 2 * Math.PI); rctx.fill(); }
  }

  function readout() {
    const t1 = 1 / Math.abs(model.lam[1]), t1r = 1 / Math.abs(ref.lam[1]);
    out.innerHTML = `LITL saw <b>${nBudget.toLocaleString()}</b> random directions and their energies, and no trajectories or forces. ` +
      `Slowest relaxation time τ₁ = 1/|λ₁| = <b>${t1.toFixed(2)}</b> (reference ${t1r.toFixed(2)}, error ${(100 * Math.abs(t1 - t1r) / t1r).toFixed(1)}%). ` +
      `Uniaxial barrier K<sub>u</sub>/k<sub>B</sub>T = ${(Ku / kT).toFixed(1)}.`;
  }

  // ------------------------------------------------------------------ refit pipeline
  let timer = null;
  function refit() {
    ref = fit(fibPts, NF); addNorth(ref);
    model = fit(uniformSamples(nBudget), nBudget); addNorth(model);
    fillGrids(model);
    Tmax = Math.min(150, Math.max(2, 4 / Math.abs(ref.lam[1])));
    renderSphere(); drawSpectrum(); readout(); saturate();
  }
  function schedule() { out.innerHTML = "refitting…"; clearTimeout(timer); timer = setTimeout(refit, 120); }

  // ------------------------------------------------------------------ controls
  const bind = (id, set, fmt) => { const el = q(id), o = q(id + "-out"); el.addEventListener("input", () => { set(parseFloat(el.value)); o.textContent = fmt(parseFloat(el.value)); schedule(); }); };
  bind("#mag-ku", (v) => (Ku = v), (v) => v.toFixed(2));
  bind("#mag-kc", (v) => (Kc = v), (v) => v.toFixed(2));
  bind("#mag-kt", (v) => (kT = v), (v) => v.toFixed(2));
  const seg = (sel, attr, fn) => root.querySelectorAll(sel).forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll(sel).forEach((x) => x.classList.toggle("on", x === b)); fn(b.dataset[attr]);
  }));
  seg("[data-n]", "n", (v) => { nBudget = parseInt(v, 10); schedule(); });
  seg("[data-color]", "color", (v) => { colorBy = parseInt(v, 10); renderSphere(); drawSpectrum(); });
  seg("[data-force]", "force", (v) => { useLearned = v === "litl"; saturate(); });
  q("#mag-resample").addEventListener("click", () => { seed = (seed * 7919 + 13) | 0; schedule(); });
  q("#mag-saturate").addEventListener("click", saturate);
  const pb = q("#mag-pause"); pb.addEventListener("click", () => { paused = !paused; pb.textContent = paused ? "Play" : "Pause"; });

  // drag to rotate
  let drag = null;
  over.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; over.setPointerCapture(e.pointerId); });
  over.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY];
    R = mul(rotX(dy * 0.01), mul(rotY(dx * 0.01), R)); renderSphere(); drawSwarm();
  });
  const end = () => { drag = null; };
  over.addEventListener("pointerup", end); over.addEventListener("pointercancel", end);

  // ------------------------------------------------------------------ loop
  let visible = true;
  if ("IntersectionObserver" in window) new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(root);
  let last = null;
  function frame(now) {
    const el = last === null ? 1 / 60 : Math.min(0.5, (now - last) / 1000); last = now;
    if (model && !paused && visible) {
      const simPerSec = Math.min(40, Math.max(0.6, Tmax / 7));
      const dt = Math.min(0.01, 0.08 * kT), n = Math.max(1, Math.round(simPerSec * el / dt));
      for (let k = 0; k < n; k++) step(dt);
      simT += n * dt;
      if (relaxOn) { relax.push([simT, meanMz()]); if (simT > Tmax) relaxOn = false; }
      drawSwarm(); drawRelax();
    }
    requestAnimationFrame(frame);
  }
  refit();
  requestAnimationFrame(frame);
})();
