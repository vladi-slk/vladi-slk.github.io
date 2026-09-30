// Interactive: which harmonics of the Duffing limit cycle does a rank-r estimator of F(L) learn?
// On the limit cycle the generator is skew-adjoint with eigenvalues lambda = i k (k = +-1, +-2, ...).
// A filter F keeps the eigenfunctions and re-weights the eigenvalues to F(i k); a rank-r estimator
// targets the r eigenvalues of F(L) with the largest modulus (its "dominant spectrum").
(function () {
  const root = document.getElementById("tk-explorer");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const dt = 0.1, KMAX = 31;                                   // Nyquist: pi/dt = 31.4 rad/s
  const C = { keep: "#eb6834", drop: "#d5d9de", tie: "#f3b597", ink: "#1f2328", ink2: "#57606a", grid: "#e6e8eb", energy: "#2a78d6" };

  // share of Var(x) carried by harmonic k on the simulated limit cycle (least-squares fit, k = 1..31);
  // values below 1e-8 are numerical noise and are treated as zero
  const SHARE = [9.990e-01, 0, 1.009e-03, 0, 1.112e-06];
  const share = (k) => (SHARE[Math.abs(k) - 1] || 0) / 2;    // split equally between +k and -k

  // band-limited pseudo-inverse, 0.1-1 Hz: a_j = -(1/pi) int sin(j phi)/phi, Jackson-type damping
  const ELL = 200, phi1 = 2 * Math.PI * 0.1 * dt, phi2 = 2 * Math.PI * 1.0 * dt;
  const aj = new Float64Array(ELL + 1);
  (function () {
    const N = 4000, h = (phi2 - phi1) / N;
    for (let j = 1; j <= ELL; j++) {
      let s = 0;
      for (let q = 0; q <= N; q++) {
        const p = phi1 + q * h, w = q === 0 || q === N ? 1 : q % 2 ? 4 : 2;
        s += (w * Math.sin(j * p)) / p;
      }
      aj[j] = (-(s * h) / 3 / Math.PI) * (1 - j / (ELL + 1));
    }
  })();
  const band = (k) => { let s = 0; for (let j = 1; j <= ELL; j++) s += aj[j] * Math.sin(j * k * dt); return Math.abs(2 * s); };

  const FILTERS = {
    koop: { F: () => 1,
      note: "T(z) = z. Every harmonic gets weight |exp(ikΔt)| = 1, so all of them tie. The filter expresses no preference, and which modes survive the rank cut is decided by noise and by the features, not by the physics." },
    sinh: { F: (k) => Math.abs(Math.sin(k * dt)),
      note: "T(z) = (z − 1/z)/2 gives weight |sin(kΔt)|. Its estimated eigenvalues are purely imaginary by construction, but the loudest harmonics sit near half the Nyquist frequency, where the Duffing signal has almost no energy." },
    res: { F: (k) => 1 / Math.hypot(0.25, theta - k),
      note: "Laplace-transform weights a_j ≈ Δt·exp(−μ jΔt) give weight 1/|μ − ik| with μ = 0.25 + iθ: a zoom lens on the harmonics near θ. It is not symmetric in ±k, so move the lens and watch which side wins." },
    band: { F: (k) => band(k),
      note: "Sine-integral weights, damped against the Gibbs effect, give weight ≈ 1/(|k|Δt) inside 0.1–1 Hz (|k| ≤ 6) and ≈ 0 outside. Its estimated eigenvalues are purely imaginary by construction, and the energetic low harmonics dominate." },
  };

  let filter = "koop", rank = 10, theta = 1;
  const svg = root.querySelector("#tk-chart"), note = root.querySelector("#tk-note"), out = root.querySelector("#tk-out");
  const rIn = root.querySelector("#tk-rank"), rOut = root.querySelector("#tk-rank-out");
  const thWrap = root.querySelector("#tk-theta-wrap"), thIn = root.querySelector("#tk-theta"), thOut = root.querySelector("#tk-theta-out");
  const el = (tag, a, p) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const txt = (s, a, p) => { const t = el("text", a, p); t.textContent = s; return t; };
  const ks = []; for (let k = -KMAX; k <= KMAX; k++) if (k) ks.push(k);

  function draw() {
    const F = FILTERS[filter].F;
    const w = ks.map((k) => F(k)), wmax = Math.max(...w);
    const tie = Math.max(...w) - Math.min(...w) < 1e-9;
    const order = ks.map((k, i) => ({ k, v: w[i] / wmax })).sort((a, b) => b.v - a.v || Math.abs(a.k) - Math.abs(b.k));
    const kept = new Set(tie ? [] : order.slice(0, rank).map((o) => o.k));

    svg.innerHTML = "";
    const W = 620, L = 46, R = 10, T1 = 34, H1 = 130, T2 = T1 + H1 + 50, H2 = 90, H = T2 + H2 + 46;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const slot = (W - L - R) / ks.length, X = (k) => L + slot * (ks.indexOf(k) + 0.5), bw = Math.max(3, slot * 0.66);
    // panel 1: filter weights
    txt("filter weight |F(ik)|, normalised" + (tie ? "  (all equal: a tie)" : "  (orange = kept at rank " + rank + ")"),
      { x: L, y: T1 - 12, "font-size": 16, fill: C.ink2 }, svg);
    for (const v of [0, 0.5, 1]) {
      el("line", { x1: L, x2: W - R, y1: T1 + H1 * (1 - v), y2: T1 + H1 * (1 - v), stroke: C.grid }, svg);
      txt(v.toFixed(1), { x: L - 8, y: T1 + H1 * (1 - v) + 4, "font-size": 14.5, fill: C.ink2, "text-anchor": "end" }, svg);
    }
    ks.forEach((k, i) => {
      const v = w[i] / wmax, h = Math.max(1, H1 * v);
      el("rect", { x: X(k) - bw / 2, y: T1 + H1 - h, width: bw, height: h, rx: 2, fill: tie ? C.tie : kept.has(k) ? C.keep : C.drop }, svg);
    });
    // panel 2: where the signal is
    txt("share of the variance of x(t) in each harmonic (log scale)", { x: L, y: T2 - 12, "font-size": 16, fill: C.ink2 }, svg);
    const Y2 = (s) => T2 + H2 * (1 - (Math.log10(s) + 8) / 8);
    for (const e of [0, -4, -8]) {
      el("line", { x1: L, x2: W - R, y1: Y2(10 ** e), y2: Y2(10 ** e), stroke: C.grid }, svg);
      txt(e === 0 ? "1" : "1e" + e, { x: L - 8, y: Y2(10 ** e) + 4, "font-size": 14.5, fill: C.ink2, "text-anchor": "end" }, svg);
    }
    ks.forEach((k) => {
      const s = share(k);
      if (s < 1e-8) return;
      el("rect", { x: X(k) - bw / 2, y: Y2(s), width: bw, height: T2 + H2 - Y2(s), rx: 2,
        fill: C.energy, opacity: tie || kept.has(k) ? 1 : 0.3 }, svg);
    });
    txt("≈ 0 elsewhere", { x: X(12), y: T2 + H2 - 8, "font-size": 14.5, fill: C.ink2 }, svg);
    for (const k of [-30, -20, -10, -3, 3, 10, 20, 30])
      txt(String(k), { x: X(k), y: T2 + H2 + 20, "font-size": 14.5, fill: C.ink2, "text-anchor": "middle" }, svg);
    txt("±1", { x: (X(-1) + X(1)) / 2, y: T2 + H2 + 20, "font-size": 14.5, fill: "#57606a", "text-anchor": "middle" }, svg);
    txt("harmonic k  (eigenvalue λ = ik rad/s)", { x: (L + W) / 2, y: H - 4, "font-size": 16, fill: C.ink2, "text-anchor": "middle" }, svg);

    note.textContent = FILTERS[filter].note;
    if (tie) {
      out.innerHTML = "Kept by a rank-" + rank + " estimator: <b>not decided by the filter</b>.";
    } else {
      const kk = [...kept].sort((a, b) => a - b), cap = kk.reduce((s, k) => s + share(k), 0);
      out.innerHTML = "Kept by a rank-" + rank + " estimator: λ = " + kk.map((k) => k + "i").join(", ") +
        " · variance of x they carry: <b>" + (cap < 1e-6 ? "≈ 0%" : (100 * cap).toFixed(cap > 0.999 ? 2 : 1) + "%") + "</b>";
    }
  }

  function update() {
    rank = parseInt(rIn.value, 10); rOut.textContent = rank;
    theta = parseFloat(thIn.value); thOut.textContent = theta.toFixed(1);
    thWrap.style.display = filter === "res" ? "" : "none";
    draw();
  }
  root.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.filter;
    root.querySelectorAll("[data-filter]").forEach((x) => x.classList.toggle("on", x === b));
    update();
  }));
  rIn.addEventListener("input", update);
  thIn.addEventListener("input", update);
  update();
})();
