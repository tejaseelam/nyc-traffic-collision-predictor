"use strict";

const DATA_FILES = {
  grid: "./data/grid.geojson",
  summary: "./data/cell_period_summary.json",
  detail: "./data/cell_detail.json",
  metrics: "./data/metrics.json",
  subgroups: "./data/subgroup_metrics.json",
};

const MODEL_INFO = {
  baseline: { name: "Rolling cell + time rate", key: "b", color: "#d66a2c" },
  logistic: { name: "Logistic regression", key: "l", color: "#14807e" },
  tree: { name: "LightGBM + location", key: "t", color: "#6d5aa8" },
};

const RAMPS = {
  observed: ["#fff7ed", "#fed7aa", "#fb923c", "#c2410c", "#7c2d12"],
  predicted: ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#172554"],
  difference: ["#b35806", "#f1a340", "#f7f7f7", "#998ec3", "#542788"],
};

const state = {
  view: "predicted",
  model: "tree",
  hour: 18,
  month: 0,
  borough: "ALL",
  weather: "all",
  topK: 5,
  compare: false,
  selectedIndex: null,
};

const appData = {};
const mapViews = [];
const tooltip = document.querySelector("#tooltip");
const statusBanner = document.querySelector("#status-banner");

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function formatPercent(value, digits = 1) {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}
function formatNumber(value, digits = 0) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";
}
function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
}
function rgbToHex(rgb) {
  return `#${rgb.map((part) => Math.round(part).toString(16).padStart(2, "0")).join("")}`;
}
function interpolateRamp(ramp, t) {
  const scaled = clamp(t, 0, 1) * (ramp.length - 1);
  const index = Math.min(ramp.length - 2, Math.floor(scaled));
  const local = scaled - index;
  const left = hexToRgb(ramp[index]);
  const right = hexToRgb(ramp[index + 1]);
  return rgbToHex(left.map((value, i) => value + (right[i] - value) * local));
}

async function fetchJson(label, url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${label} (${response.status})`);
  return response.json();
}

function showStatus(message, error = false) {
  statusBanner.hidden = false;
  statusBanner.classList.toggle("error", error);
  statusBanner.textContent = message;
}

function flattenCoordinates(geometry) {
  if (geometry.type === "Polygon") return geometry.coordinates.flat(1);
  if (geometry.type === "MultiPolygon") return geometry.coordinates.flat(2);
  return [];
}

function geometryPath(geometry, project) {
  const polygonPath = (polygon) => polygon.map((ring) => {
    const points = ring.map(([lon, lat]) => project(lon, lat));
    return points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join("") + "Z";
  }).join("");
  if (geometry.type === "Polygon") return polygonPath(geometry.coordinates);
  if (geometry.type === "MultiPolygon") return geometry.coordinates.map(polygonPath).join("");
  return "";
}

class GridMap {
  constructor(svg, role) {
    this.svg = svg;
    this.role = role;
    this.width = 700;
    this.height = 820;
    this.scale = 1;
    this.x = 0;
    this.y = 0;
    this.dragging = false;
    this.pointer = null;
    this.svg.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);
    this.group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    this.svg.append(this.group);
    this.attachPanZoom();
  }

  build(features) {
    const coordinates = features.flatMap((feature) => flattenCoordinates(feature.geometry));
    const lons = coordinates.map((point) => point[0]);
    const lats = coordinates.map((point) => point[1]);
    const bounds = {
      minLon: Math.min(...lons), maxLon: Math.max(...lons),
      minLat: Math.min(...lats), maxLat: Math.max(...lats),
    };
    const pad = 28;
    const xScale = (this.width - pad * 2) / (bounds.maxLon - bounds.minLon);
    const yScale = (this.height - pad * 2) / (bounds.maxLat - bounds.minLat);
    const scale = Math.min(xScale, yScale);
    const usedWidth = (bounds.maxLon - bounds.minLon) * scale;
    const usedHeight = (bounds.maxLat - bounds.minLat) * scale;
    const xOffset = (this.width - usedWidth) / 2;
    const yOffset = (this.height - usedHeight) / 2;
    const project = (lon, lat) => [
      xOffset + (lon - bounds.minLon) * scale,
      yOffset + (bounds.maxLat - lat) * scale,
    ];

    const fragment = document.createDocumentFragment();
    this.paths = features.map((feature, index) => {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", geometryPath(feature.geometry, project));
      path.setAttribute("class", "grid-cell");
      path.dataset.index = index;
      path.addEventListener("mouseenter", (event) => showTooltip(event, index, this.role));
      path.addEventListener("mousemove", positionTooltip);
      path.addEventListener("mouseleave", () => { tooltip.hidden = true; });
      path.addEventListener("click", () => selectCell(index));
      fragment.append(path);
      return path;
    });
    this.group.append(fragment);
  }

  render(values, view, topCells, visible) {
    const legend = appData.summary.meta.legend;
    const max = view === "observed" ? legend.observed_max : legend.predicted_max;
    const diffMax = legend.difference_abs_max;
    this.paths.forEach((path, index) => {
      const isVisible = visible(index);
      path.style.display = isVisible ? "" : "none";
      if (!isVisible) return;
      const value = values[index];
      const t = view === "difference"
        ? (value + diffMax) / (2 * diffMax)
        : value / max;
      path.setAttribute("fill", interpolateRamp(RAMPS[view], t));
      path.classList.toggle("top-cell", topCells.has(index));
      path.classList.toggle("selected", state.selectedIndex === index);
    });
  }

  attachPanZoom() {
    this.svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const next = clamp(this.scale * (event.deltaY < 0 ? 1.15 : 0.87), 1, 8);
      const box = this.svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * this.width;
      const py = ((event.clientY - box.top) / box.height) * this.height;
      this.x = px - ((px - this.x) * next) / this.scale;
      this.y = py - ((py - this.y) * next) / this.scale;
      this.scale = next;
      this.applyTransform(true);
    }, { passive: false });
    this.svg.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.pointer = { x: event.clientX, y: event.clientY, originX: this.x, originY: this.y };
      this.svg.setPointerCapture(event.pointerId);
    });
    this.svg.addEventListener("pointermove", (event) => {
      if (!this.dragging) return;
      const box = this.svg.getBoundingClientRect();
      this.x = this.pointer.originX + ((event.clientX - this.pointer.x) / box.width) * this.width;
      this.y = this.pointer.originY + ((event.clientY - this.pointer.y) / box.height) * this.height;
      this.applyTransform(true);
    });
    const stop = () => { this.dragging = false; };
    this.svg.addEventListener("pointerup", stop);
    this.svg.addEventListener("pointercancel", stop);
  }

  applyTransform(sync) {
    this.group.setAttribute("transform", `translate(${this.x} ${this.y}) scale(${this.scale})`);
    if (sync && state.compare) {
      mapViews.filter((map) => map !== this).forEach((map) => {
        map.x = this.x; map.y = this.y; map.scale = this.scale; map.applyTransform(false);
      });
    }
  }

  reset() {
    this.scale = 1; this.x = 0; this.y = 0; this.applyTransform(true);
  }
}

function currentSlice() {
  const key = `${state.month}-${state.hour}`;
  return appData.summary.slices[state.weather]?.[key] || appData.summary.slices.all[key];
}

function decoded(array) {
  const scale = appData.summary.meta.scale;
  return array.map((value) => value / scale);
}

function mapValues(view, model, slice) {
  const observed = decoded(slice.o);
  if (view === "observed") return observed;
  const predicted = decoded(slice[MODEL_INFO[model].key]);
  if (view === "predicted") return predicted;
  return predicted.map((value, index) => value - observed[index]);
}

function visibleCell(index) {
  const borough = appData.grid.features[index].properties.borough;
  return state.borough === "ALL" || borough === state.borough;
}

function topCellsFor(slice) {
  const predicted = decoded(slice[MODEL_INFO[state.model].key]);
  const eligible = predicted.map((value, index) => ({ value, index })).filter((item) => visibleCell(item.index));
  eligible.sort((a, b) => b.value - a.value);
  return new Set(eligible.slice(0, Math.max(1, Math.ceil(eligible.length * state.topK / 100))).map((item) => item.index));
}

function updateExplorer() {
  const slice = currentSlice();
  if (!slice) return;
  const topCells = topCellsFor(slice);
  const predictedValues = mapValues("predicted", state.model, slice);
  const observedValues = mapValues("observed", state.model, slice);
  const primaryView = state.compare ? "predicted" : state.view;
  const primaryValues = state.compare ? predictedValues : mapValues(state.view, state.model, slice);
  mapViews[0].render(primaryValues, primaryView, topCells, visibleCell);
  if (state.compare) mapViews[1].render(observedValues, "observed", topCells, visibleCell);

  const totalCrashes = slice.c.reduce((sum, value, index) => sum + (visibleCell(index) ? value : 0), 0);
  const captured = slice.c.reduce((sum, value, index) => sum + (visibleCell(index) && topCells.has(index) ? value : 0), 0);
  document.querySelector("#capture-value").textContent = totalCrashes ? formatPercent(captured / totalCrashes) : "No crashes";
  document.querySelector("#top-k-output").textContent = `Top ${state.topK}%`;

  const viewLabel = state.compare ? "Predicted vs. observed" : {
    predicted: "Predicted collision risk", observed: "Observed crash rate", difference: "Prediction minus observed",
  }[state.view];
  document.querySelector("#map-title").textContent = viewLabel;
  document.querySelector("#map-subtitle").textContent = `${MODEL_INFO[state.model].name} · 2023 validation`;
  document.querySelector("#primary-map-label").textContent = state.compare ? "Predicted" : viewLabel.replace(" collision risk", "").replace(" crash rate", "");

  const badge = document.querySelector("#view-badge");
  const badgeType = state.compare ? "predicted" : state.view;
  badge.className = `view-badge ${badgeType}`;
  badge.textContent = state.compare ? "Separate recorded and estimated views" : {
    predicted: "Model estimate", observed: "Recorded crashes", difference: "Estimate − recorded",
  }[state.view];
  renderLegend(primaryView);
  if (state.selectedIndex !== null) renderCellDetail(state.selectedIndex);
}

function renderLegend(view) {
  const legend = document.querySelector("#map-legend");
  const meta = appData.summary.meta.legend;
  const max = view === "observed" ? meta.observed_max : meta.predicted_max;
  const labels = view === "difference"
    ? [formatPercent(-meta.difference_abs_max), "0", `+${formatPercent(meta.difference_abs_max)}`]
    : ["0", formatPercent(max / 2), formatPercent(max)];
  legend.innerHTML = `<div class="legend-scale">${RAMPS[view].map((color) => `<span style="background:${color}"></span>`).join("")}</div><div class="legend-labels"><span>${labels[0]}</span><span>${labels[1]}</span><span>${labels[2]}</span></div>`;
}

function showTooltip(event, index, role) {
  const feature = appData.grid.features[index];
  const slice = currentSlice();
  const risk = decoded(slice[MODEL_INFO[state.model].key])[index];
  const observed = decoded(slice.o)[index];
  tooltip.innerHTML = `<strong>${feature.properties.cell_id}</strong><br>${feature.properties.borough}<br>Predicted: ${formatPercent(risk, 2)}<br>Observed: ${formatPercent(observed, 2)}`;
  tooltip.hidden = false;
  positionTooltip(event);
}

function positionTooltip(event) {
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
}

function selectCell(index) {
  state.selectedIndex = index;
  renderCellDetail(index);
  updateExplorer();
}

function renderCellDetail(index) {
  const feature = appData.grid.features[index];
  const props = feature.properties;
  const slice = currentSlice();
  const risk = decoded(slice[MODEL_INFO[state.model].key])[index];
  const observed = decoded(slice.o)[index];
  document.querySelector("#empty-detail").hidden = true;
  document.querySelector("#cell-detail").hidden = false;
  document.querySelector("#detail-borough").textContent = props.borough;
  document.querySelector("#detail-cell-id").textContent = props.cell_id;
  document.querySelector("#detail-risk").textContent = formatPercent(risk, 2);
  document.querySelector("#detail-observed").textContent = formatPercent(observed, 2);
  const percentile = appData.detail.percentile[state.model][index] / 10;
  document.querySelector("#detail-percentile").textContent = `${percentile.toFixed(1)}th percentile`;
  renderSparkline(appData.detail.monthly[index]);
  renderContext(index);
  document.querySelector("#cell-facts").innerHTML = `
    <dt>Land share</dt><dd>${formatPercent(props.land_fraction, 0)}</dd>
    <dt>Mapped road length</dt><dd>${formatNumber(props.road_length_m)} m</dd>
    <dt>Average speed limit</dt><dd>${props.speed_limit_mph ? `${formatNumber(props.speed_limit_mph)} mph` : "Unavailable"}</dd>
    <dt>Crashes in current filter</dt><dd>${formatNumber(slice.c[index])}</dd>`;
}

function renderSparkline(values) {
  const svg = document.querySelector("#history-sparkline");
  const width = 250; const height = 86; const pad = 8;
  const max = Math.max(1, ...values);
  const points = values.map((value, index) => [pad + index * (width - pad * 2) / (values.length - 1), height - pad - value / max * (height - pad * 2)]);
  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L${points.at(-1)[0]},${height - pad} L${points[0][0]},${height - pad} Z`;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `<path d="${area}" fill="#d8eeea"></path><path d="${line}" fill="none" stroke="#14807e" stroke-width="2"></path><line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#a9b5b2"></line>`;
}

function renderContext(index) {
  const context = appData.detail.context;
  const items = [
    ["Crash history", context.history[index]],
    ["Time pattern", context.time[index]],
    ["Roads / speed", context.roads[index]],
  ];
  document.querySelector("#context-bars").innerHTML = items.map(([label, value]) => `
    <div class="context-row"><span>${label}</span><div class="context-track"><div class="context-fill" style="width:${clamp(value / 2, 0, 1) * 100}%"></div></div><strong>${value.toFixed(1)}×</strong></div>`).join("");
}

function switchTab(tab) {
  document.querySelectorAll(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach((panel) => { panel.hidden = panel.dataset.panel !== tab; });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function bindControls() {
  document.querySelectorAll(".nav-link").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  document.querySelectorAll("[data-tab-jump]").forEach((button) => button.addEventListener("click", () => switchTab(button.dataset.tabJump)));
  document.querySelectorAll("#view-toggle button").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.value;
    document.querySelectorAll("#view-toggle button").forEach((item) => item.classList.toggle("active", item === button));
    updateExplorer();
  }));
  const selects = [
    ["model-select", "model", String], ["hour-select", "hour", Number], ["month-select", "month", Number],
    ["borough-select", "borough", String], ["weather-select", "weather", String],
  ];
  selects.forEach(([id, key, converter]) => document.querySelector(`#${id}`).addEventListener("change", (event) => { state[key] = converter(event.target.value); updateExplorer(); }));
  document.querySelector("#top-k").addEventListener("input", (event) => { state.topK = Number(event.target.value); updateExplorer(); });
  document.querySelector("#compare-toggle").addEventListener("change", (event) => {
    state.compare = event.target.checked;
    document.querySelector("#map-grid").classList.toggle("comparing", state.compare);
    document.querySelector("[data-map-role='comparison']").hidden = !state.compare;
    updateExplorer();
  });
  document.querySelector("#reset-map").addEventListener("click", () => mapViews[0].reset());
  document.querySelector("#presentation-button").addEventListener("click", (event) => {
    const active = document.body.classList.toggle("presentation");
    event.currentTarget.setAttribute("aria-pressed", active);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "p" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
      document.querySelector("#presentation-button").click();
    }
    if (event.key === "Escape" && document.body.classList.contains("presentation")) document.querySelector("#presentation-button").click();
  });
  document.querySelectorAll("[data-story]").forEach((button) => button.addEventListener("click", () => runStory(button.dataset.story)));
  document.querySelector("#subgroup-select").addEventListener("change", (event) => renderSubgroups(event.target.value));
}

function syncControls() {
  document.querySelector("#model-select").value = state.model;
  document.querySelector("#hour-select").value = String(state.hour);
  document.querySelector("#month-select").value = String(state.month);
  document.querySelector("#borough-select").value = state.borough;
  document.querySelector("#weather-select").value = state.weather;
  document.querySelector("#top-k").value = String(state.topK);
  document.querySelector("#compare-toggle").checked = state.compare;
  document.querySelector("#map-grid").classList.toggle("comparing", state.compare);
  document.querySelector("[data-map-role='comparison']").hidden = !state.compare;
  document.querySelectorAll("#view-toggle button").forEach((button) => button.classList.toggle("active", button.dataset.value === state.view));
}

function runStory(story) {
  if (story === "compare") Object.assign(state, { model: "tree", hour: 18, month: 0, weather: "all", borough: "ALL", compare: true, view: "predicted" });
  if (story === "capture") Object.assign(state, { model: "tree", hour: 18, month: 0, topK: 5, compare: false, view: "predicted" });
  if (story === "cell") {
    Object.assign(state, { model: "tree", hour: 18, month: 0, compare: false, view: "predicted" });
    const values = decoded(currentSlice().t);
    state.selectedIndex = values.reduce((best, value, index) => value > values[best] ? index : best, 0);
  }
  syncControls(); updateExplorer();
}

function renderMetrics() {
  const metrics = appData.metrics;
  const best = metrics.models.reduce((left, right) => right.average_precision > left.average_precision ? right : left);
  document.querySelector("#metric-cards").innerHTML = [
    ["Best average precision", best.average_precision.toFixed(4), best.name],
    ["Top-5% capture", formatPercent(best.top5_capture), best.name],
    ["Validation positives", formatPercent(metrics.validation.positive_rate, 2), `${formatNumber(metrics.validation.positive_rows)} crash cell-periods`],
    ["Grid × time panel", formatNumber(metrics.validation.rows), "3,640 cells · six-hour periods"],
  ].map(([label, value, note]) => `<article class="metric-card"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("");

  document.querySelector("#model-table").innerHTML = `<thead><tr><th>Model</th><th>Average precision</th><th>ROC AUC</th><th>Top-5% capture</th><th>Brier score</th></tr></thead><tbody>${metrics.models.map((model) => `<tr class="${model.id === best.id ? "best" : ""}"><td>${model.name}</td><td>${model.average_precision.toFixed(4)}</td><td>${model.roc_auc.toFixed(3)}</td><td>${formatPercent(model.top5_capture)}</td><td>${model.brier_score.toFixed(4)}</td></tr>`).join("")}</tbody>`;
  drawLineChart("capture-chart", metrics.curves.capture, { xLabel: "Highest-risk share of cells", yLabel: "Share of crashes captured", diagonal: true });
  drawLineChart("pr-chart", metrics.curves.precision_recall, { xLabel: "Recall", yLabel: "Precision" });
  drawLineChart("calibration-chart", metrics.curves.calibration, { xLabel: "Predicted probability", yLabel: "Observed rate", diagonal: true });
  const maxDrop = Math.max(...metrics.importance.map((item) => Math.max(0, item.drop)));
  document.querySelector("#importance-chart").innerHTML = metrics.importance.map((item) => `<div class="importance-row"><span>${item.name}</span><div class="importance-track"><div class="importance-fill" style="width:${Math.max(0, item.drop) / maxDrop * 100}%"></div></div><strong>${item.drop.toFixed(3)}</strong></div>`).join("");
  renderSubgroups("borough");
}

function drawLineChart(id, series, options) {
  const svg = document.querySelector(`#${id}`);
  const width = 520; const height = 280; const margin = { left: 50, right: 18, top: 14, bottom: 42 };
  const all = Object.values(series).flat();
  const xMax = Math.max(...all.map((point) => point[0]), .01);
  const yMax = Math.max(...all.map((point) => point[1]), .01) * 1.05;
  const x = (value) => margin.left + value / xMax * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - value / yMax * (height - margin.top - margin.bottom);
  const grid = [0, .25, .5, .75, 1].map((fraction) => `<line class="grid-line" x1="${margin.left}" y1="${y(yMax * fraction)}" x2="${width - margin.right}" y2="${y(yMax * fraction)}"></line><text class="axis-label" x="${margin.left - 7}" y="${y(yMax * fraction) + 3}" text-anchor="end">${(yMax * fraction).toFixed(2)}</text>`).join("");
  const paths = Object.entries(series).map(([model, points]) => {
    const path = points.map(([px, py], index) => `${index ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`).join(" ");
    return `<path d="${path}" fill="none" stroke="${MODEL_INFO[model].color}" stroke-width="2.6"></path>`;
  }).join("");
  const diagonal = options.diagonal ? `<line x1="${x(0)}" y1="${y(0)}" x2="${x(Math.min(xMax, yMax))}" y2="${y(Math.min(xMax, yMax))}" stroke="#9ca7a4" stroke-dasharray="4 4"></line>` : "";
  const legend = Object.keys(series).map((model, index) => `<g transform="translate(${margin.left + index * 135},${height - 8})"><line x1="0" y1="0" x2="18" y2="0" stroke="${MODEL_INFO[model].color}" stroke-width="3"></line><text class="axis-label" x="23" y="3">${MODEL_INFO[model].name.replace(" + location", "")}</text></g>`).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${grid}${diagonal}<line class="axis-line" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"></line><line class="axis-line" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"></line>${paths}<text class="axis-label" x="${width / 2}" y="${height - 24}" text-anchor="middle">${options.xLabel}</text><text class="axis-label" transform="translate(13 ${height / 2}) rotate(-90)" text-anchor="middle">${options.yLabel}</text>${legend}`;
}

function renderSubgroups(group) {
  const rows = appData.subgroups[group] || [];
  document.querySelector("#subgroup-table").innerHTML = `<thead><tr><th>${group.replace(/^./, (letter) => letter.toUpperCase())}</th><th>Model</th><th>Rows</th><th>Positive rate</th><th>Average precision</th><th>Top-5% capture</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${row.group}</td><td>${MODEL_INFO[row.model].name}</td><td>${formatNumber(row.rows)}</td><td>${formatPercent(row.positive_rate, 2)}</td><td>${row.average_precision.toFixed(4)}</td><td>${formatPercent(row.top5_capture)}</td></tr>`).join("")}</tbody>`;
}

async function init() {
  showStatus("Loading the 2023 validation explorer…");
  try {
    const [grid, summary, detail, metrics, subgroups] = await Promise.all(Object.entries(DATA_FILES).map(([label, url]) => fetchJson(label, url)));
    Object.assign(appData, { grid, summary, detail, metrics, subgroups });
    statusBanner.hidden = true;
    mapViews.push(new GridMap(document.querySelector("#primary-map"), "primary"));
    mapViews.push(new GridMap(document.querySelector("#comparison-map"), "comparison"));
    mapViews.forEach((map) => map.build(grid.features));
    bindControls();
    syncControls();
    updateExplorer();
    renderMetrics();
  } catch (error) {
    console.error(error);
    showStatus(`The explorer could not load a required data file: ${error.message}. Run the dashboard export script before publishing.`, true);
  }
}

init();
