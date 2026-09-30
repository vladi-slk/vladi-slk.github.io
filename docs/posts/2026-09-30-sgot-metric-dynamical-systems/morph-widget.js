// Interactive: interpolating two oscillators by (a) linear averaging of transfer operators
// (the Hilbert–Schmidt barycenter) and (b) the SGOT barycenter (straight line in decay/frequency).
(function () {
  const root = document.getElementById("sgot-morph");
  if (!root) return;
  const NS = "http://www.w3.org/2000/svg";
  const C = { A: "#2a78d6", B: "#1baf7a", hs: "#5f6368", sgot: "#eb6834", ink: "#1f2328", ink2: "#57606a", grid: "#e6e8eb" };
  // systems: generator eigenvalue lambda = -decay + i 2 pi f  (the conjugate is implied)
  const sysA = { f: 0.2, d: 0.03 }, sysB = { f: 0.8, d: 0.25 };

  const gIn = root.querySelector("#sm-gamma"), gOut = root.querySelector("#sm-gamma-out");
  const fsBtns = root.querySelectorAll("[data-fs]");
  const disc = root.querySelector("#sm-disc"), sig = root.querySelector("#sm-sig");
  const tbody = root.querySelector("#sm-readout tbody");
  let fs = 2;

  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const txt = (s, attrs, parent) => { const t = el("text", attrs, parent); t.textContent = s; return t; };

  // complex helpers on [re, im]
  const cexp = (re, im) => [Math.exp(re) * Math.cos(im), Math.exp(re) * Math.sin(im)];
  const toNu = (s, dt) => cexp(-s.d * dt, 2 * Math.PI * s.f * dt);          // transfer-operator eigenvalue
  const fromNu = (nu, dt) => ({                                              // back to physical units
    d: -Math.log(Math.hypot(nu[0], nu[1])) / dt,
    f: Math.atan2(nu[1], nu[0]) / (2 * Math.PI * dt),
  });
  const hsMix = (g, dt) => { const a = toNu(sysA, dt), b = toNu(sysB, dt); return [(1 - g) * a[0] + g * b[0], (1 - g) * a[1] + g * b[1]]; };
  const sgotMix = (g) => ({ f: (1 - g) * sysA.f + g * sysB.f, d: (1 - g) * sysA.d + g * sysB.d });

  function drawDisc(g) {
    disc.innerHTML = "";
    const S = 320, cx = 160, cy = 160, R = 130;
    const X = (re) => cx + R * re, Y = (im) => cy - R * im;
    el("line", { x1: 20, y1: cy, x2: 300, y2: cy, stroke: C.grid }, disc);
    el("line", { x1: cx, y1: 20, x2: cx, y2: 300, stroke: C.grid }, disc);
    el("circle", { cx, cy, r: R, fill: "none", stroke: "#c9ced4", "stroke-width": 1.2 }, disc);
    txt("unit circle = undamped", { x: cx, y: 312, "font-size": 12.5, fill: C.ink2, "text-anchor": "middle" }, disc);
    const dt = 1 / fs;
    // SGOT path (curve) and HS path (chord)
    let dS = "", dH = "";
    for (let i = 0; i <= 100; i++) {
      const gg = i / 100, nu = toNu(sgotMix(gg), dt), h = hsMix(gg, dt);
      dS += (i ? "L" : "M") + X(nu[0]).toFixed(1) + "," + Y(nu[1]).toFixed(1);
      dH += (i ? "L" : "M") + X(h[0]).toFixed(1) + "," + Y(h[1]).toFixed(1);
    }
    el("path", { d: dH, fill: "none", stroke: C.hs, "stroke-width": 2, "stroke-dasharray": "5 4" }, disc);
    el("path", { d: dS, fill: "none", stroke: C.sgot, "stroke-width": 2.5 }, disc);
    const dot = (p, col, r, lab, dx, dy) => {
      el("circle", { cx: X(p[0]), cy: Y(p[1]), r, fill: col, stroke: "#fff", "stroke-width": 2 }, disc);
      el("circle", { cx: X(p[0]), cy: Y(-p[1]), r: r - 1.5, fill: col, opacity: 0.25 }, disc); // conjugate
      if (lab) txt(lab, { x: X(p[0]) + dx, y: Y(p[1]) + dy, "font-size": 15, fill: C.ink, "font-weight": 600 }, disc);
    };
    dot(toNu(sysA, dt), C.A, 6, "A", 8, -6);
    dot(toNu(sysB, dt), C.B, 6, "B", -16, -8);
    dot(hsMix(g, dt), C.hs, 6.5, "", 0, 0);
    dot(toNu(sgotMix(g), dt), C.sgot, 7, "", 0, 0);
    txt("transfer-operator eigenvalues  ν = exp(λΔt)", { x: 10, y: 17, "font-size": 13, fill: C.ink2 }, disc);
  }

  function drawSignals(g) {
    sig.innerHTML = "";
    const W = 540, H = 320, L = 40, Rm = 12, T = 28, B = 36, tmax = 12;
    const X = (t) => L + (W - L - Rm) * t / tmax, Y = (v) => T + (H - T - B) * (1 - v) / 2;
    for (let v = -1; v <= 1; v += 0.5) el("line", { x1: L, x2: W - Rm, y1: Y(v), y2: Y(v), stroke: C.grid }, sig);
    for (let t = 0; t <= tmax; t += 2) {
      txt(String(t), { x: X(t), y: H - B + 16, "font-size": 12.5, fill: C.ink2, "text-anchor": "middle" }, sig);
    }
    txt("time (s)", { x: (L + W) / 2, y: H - 3, "font-size": 12.5, fill: C.ink2, "text-anchor": "middle" }, sig);
    txt("impulse response of the interpolated system", { x: L, y: 17, "font-size": 13.5, fill: C.ink2 }, sig);
    const dt = 1 / fs;
    const curve = (s, col, w, dash, op) => {
      let d = "";
      for (let i = 0; i <= 600; i++) {
        const t = tmax * i / 600, v = Math.exp(-s.d * t) * Math.cos(2 * Math.PI * s.f * t);
        d += (i ? "L" : "M") + X(t).toFixed(1) + "," + Y(v).toFixed(1);
      }
      el("path", { d, fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash || "none", opacity: op || 1 }, sig);
    };
    curve(sysA, C.A, 1.3, null, 0.35);
    curve(sysB, C.B, 1.3, null, 0.35);
    curve(fromNu(hsMix(g, dt), dt), C.hs, 2, "5 4");
    curve(sgotMix(g), C.sgot, 2.6);
  }

  function readout(g) {
    const dt = 1 / fs, h = fromNu(hsMix(g, dt), dt), s = sgotMix(g);
    const rows = [["System A", C.A, sysA], ["System B", C.B, sysB], ["Linear average of operators (Hilbert–Schmidt)", C.hs, h], ["SGOT barycenter", C.sgot, s]];
    tbody.innerHTML = rows.map(([n, c, v]) =>
      `<tr><td><span class="kv-swatch" style="background:${c}"></span>${n}</td><td>${Math.abs(v.f).toFixed(3)}</td><td>${v.d.toFixed(3)}</td></tr>`).join("");
  }

  function update() {
    const g = parseFloat(gIn.value);
    gOut.textContent = g.toFixed(2);
    drawDisc(g); drawSignals(g); readout(g);
  }
  gIn.addEventListener("input", update);
  fsBtns.forEach((b) => b.addEventListener("click", () => {
    fs = parseFloat(b.dataset.fs);
    fsBtns.forEach((x) => x.classList.toggle("on", x === b));
    update();
  }));
  update();
})();
