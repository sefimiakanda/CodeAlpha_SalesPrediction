// Sales Prediction : comportement de l'interface (appels à l'API /predict, /optimize, /model/info).

const $ = (id) => document.getElementById(id);
const CH = ["tv", "radio", "newspaper"];
const NAMES = {tv: "TV", radio: "Radio", newspaper: "Newspaper"};
const COLORS = {tv: "var(--tv)", radio: "var(--radio)", newspaper: "var(--news)"};
let timer, lastBest = null;

const budget = () => Object.fromEntries(CH.map((c) => [c, +$(c + "-n").value || 0]));
const setBudget = (b) => CH.forEach((c) => { $(c).value = b[c]; $(c + "-n").value = b[c]; });
const fmt = (n, d = 1) => n.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});

CH.forEach((c) => {
  $(c).oninput = () => { $(c + "-n").value = $(c).value; refresh(); };
  $(c + "-n").oninput = () => { $(c).value = $(c + "-n").value; refresh(); };
});

function refresh() {
  const b = budget();
  $("total").textContent = fmt(b.tv + b.radio + b.newspaper) + " k$";
  clearTimeout(timer);
  timer = setTimeout(() => forecast(b), 250);
}

async function post(url, body) {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "The service is unavailable, please retry.");
  return r.json();
}

async function forecast(b) {
  try {
    const d = await post("/predict", b);
    $("sales").textContent = fmt(d.predicted_sales);
    const maxEff = Math.max(...Object.values(d.marginal_effects), 0.0001);
    $("effects").replaceChildren(...CH.map((c) => {
      const e = d.marginal_effects[NAMES[c]], row = document.createElement("div");
      row.className = "eff"; row.style.cssText = `--c:${COLORS[c]};--w:${Math.max(e / maxEff, 0) * 100}%`;
      row.innerHTML = `<div class="t"><span>${NAMES[c]}</span><span>${e < 0.0005 ? "no effect" : "+" + fmt(e * 1000, 0) + " units per extra $1k"}</span></div><div class="bar"></div>`;
      return row;
    }));
    $("warnings").replaceChildren(...d.warnings.map((w) => Object.assign(document.createElement("div"), {className: "warn", textContent: w})));
  } catch (err) {
    $("warnings").replaceChildren(Object.assign(document.createElement("div"), {className: "warn", textContent: err.message}));
  }
}

$("opt").onclick = async () => {
  const b = budget(), total = b.tv + b.radio + b.newspaper;
  if (total <= 0) { $("optres").textContent = "Set a budget above zero first."; return; }
  $("opt").disabled = true;
  try {
    const d = await post("/optimize", {total_budget: total, current: b});
    lastBest = {tv: d.allocation.TV, radio: d.allocation.Radio, newspaper: d.allocation.Newspaper};
    $("optres").innerHTML = `<div class="split"><span></span><b>Current</b><b>Recommended</b>` +
      CH.map((c) => `<span style="color:${COLORS[c]};font-weight:600">${NAMES[c]}</span><span>${fmt(b[c])} k$</span><span>${fmt(lastBest[c])} k$</span>`).join("") +
      `<b>Sales</b><span>${fmt(d.current_predicted_sales)}</span><span>${fmt(d.predicted_sales)}</span></div>` +
      `<p class="gain">${d.uplift >= 0.05 ? "+" + fmt(d.uplift) + " thousand units (" + fmt(d.uplift_pct) + " %)" : "Your split is already near-optimal."}</p>` +
      `<button id="apply">Apply this split</button> <small class="muted">${d.warnings[0]}</small>`;
    $("apply").onclick = () => { setBudget(lastBest); refresh(); };
  } catch (err) { $("optres").textContent = err.message; }
  finally { $("opt").disabled = false; }
};

fetch("/model/info").then((r) => r.json()).then((m) => {
  const r2 = m.metrics?.cross_validation?.r2_mean;
  if (r2) $("info").textContent += ` Model: linear regression with TV×Radio synergy and diminishing TV returns (cross-validated R² ${fmt(r2, 3)}, trained on ${m.n_samples} observations). Estimates are indicative.`;
}).catch(() => {});

setBudget({tv: 150, radio: 25, newspaper: 30});
refresh();
