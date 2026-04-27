/* ============================================================
   SafeLink ML Dashboard — app.js
   API_BASE points to the FastAPI server.
   ============================================================ */

const API_BASE = 'http://localhost:8000';

// ── Tiny helpers ──────────────────────────────────────────────
const $  = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);

function riskColor(level) {
  return { LOW: '#22c55e', MODERATE: '#eab308', HIGH: '#f97316', CRITICAL: '#ef4444' }[level] ?? '#64748b';
}

function magColor(m) {
  if (m >= 7) return '#9f1239';
  if (m >= 6) return '#ef4444';
  if (m >= 5) return '#f97316';
  if (m >= 4) return '#eab308';
  return '#4ade80';
}

function magClass(m) {
  if (m >= 7) return 'mag-7';
  if (m >= 6) return 'mag-6';
  if (m >= 5) return 'mag-5';
  if (m >= 4) return 'mag-4';
  return '';
}

function timeAgo(isoStr) {
  const ms = Date.now() - new Date(isoStr).getTime();
  const m  = Math.floor(ms / 60000);
  if (m < 60)   return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24)   return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function fmtCoord(v, pos, neg) {
  return `${Math.abs(v).toFixed(3)}° ${v >= 0 ? pos : neg}`;
}

// ── Tab switching ─────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b   => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${tab}`));
    if (tab === 'flood') {
      floodMap.invalidateSize();
      if (!heatmapFetched) fetchHeatmap();
    } else {
      eqMap.invalidateSize();
    }
  });
});

// ── API health ────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    if (!r.ok) throw new Error('non-200');
    const d = await r.json();
    $('status-dot').className = 'status-dot ok';
    const eq    = d.models_loaded?.earthquake ? 'loaded' : 'fallback';
    const flood = d.models_loaded?.flood      ? 'loaded' : 'fallback';
    $('status-text').textContent = `API online · EQ model: ${eq} · Flood model: ${flood}`;
  } catch {
    $('status-dot').className = 'status-dot error';
    $('status-text').textContent = 'API unreachable — run: uvicorn main:app --reload --port 8000';
  }
}

// ════════════════════════════════════════════════════════════════
//  EARTHQUAKE TAB
// ════════════════════════════════════════════════════════════════

let eqMap;
let eqOriginMarker = null, eqOriginCircle = null;
let eqMainLayer, eqAfterLayer;
let eqLat = null, eqLng = null;

function initEqMap() {
  eqMap = L.map('eq-map', { center: [20, 10], zoom: 2, worldCopyJump: true });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &amp; CARTO',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(eqMap);

  eqMainLayer  = L.layerGroup().addTo(eqMap);
  eqAfterLayer = L.layerGroup().addTo(eqMap);

  // Click on map = set origin
  eqMap.on('click', e => setEqOrigin(e.latlng.lat, e.latlng.lng));

  $('eq-search-btn').addEventListener('click', searchEarthquakes);
  $('eq-locate-btn').addEventListener('click', useMyLocation);

  // Manual coordinate inputs
  ['eq-lat', 'eq-lng'].forEach(id => {
    $(id).addEventListener('change', () => {
      const lat = parseFloat($('eq-lat').value);
      const lng = parseFloat($('eq-lng').value);
      if (!isNaN(lat) && !isNaN(lng)) setEqOrigin(lat, lng);
    });
  });
}

function setEqOrigin(lat, lng) {
  eqLat = lat; eqLng = lng;
  $('eq-lat').value = lat.toFixed(4);
  $('eq-lng').value = lng.toFixed(4);
  $('eq-search-btn').disabled = false;

  if (eqOriginMarker) { eqOriginMarker.remove(); eqOriginCircle.remove(); }

  eqOriginMarker = L.circleMarker([lat, lng], {
    radius: 7, color: '#818cf8', fillColor: '#6366f1', fillOpacity: 0.9, weight: 2,
  }).addTo(eqMap).bindPopup(`<b>Search origin</b><br>${fmtCoord(lat,'N','S')}, ${fmtCoord(lng,'E','W')}`);

  eqOriginCircle = L.circle([lat, lng], {
    radius: 200_000, color: '#6366f1', fillColor: '#6366f1',
    fillOpacity: 0.04, weight: 1.5, dashArray: '6 5',
  }).addTo(eqMap);
}

async function searchEarthquakes() {
  if (eqLat === null) return;
  const resultsEl = $('eq-results');
  resultsEl.innerHTML = `<div class="loading"><div class="spinner"></div> Querying USGS &amp; running BiLSTM model…</div>`;
  $('eq-search-btn').disabled = true;
  eqMainLayer.clearLayers();
  eqAfterLayer.clearLayers();

  try {
    const resp = await fetch(`${API_BASE}/earthquake/check`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ latitude: eqLat, longitude: eqLng, pakistan_only: false }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail ?? `HTTP ${resp.status}`);
    }
    const results = await resp.json();
    renderEqResults(results);
    plotEqOnMap(results);
  } catch (e) {
    resultsEl.innerHTML = `<div class="empty-state"><div class="empty-icon">&#9888;&#65039;</div>${e.message}</div>`;
  } finally {
    $('eq-search-btn').disabled = false;
  }
}

function renderEqResults(list) {
  const el = $('eq-results');
  if (!list.length) {
    el.innerHTML = `<div class="empty-state"><div class="empty-icon">&#128269;</div>No M3.5+ earthquakes found within 200&thinsp;km in the last 24&thinsp;h.</div>`;
    return;
  }

  const sorted = [...list].sort((a, b) => b.mainshock_magnitude - a.mainshock_magnitude);

  el.innerHTML = `<div class="results-count">${sorted.length} earthquake${sorted.length > 1 ? 's' : ''} found &mdash; click a card to see predicted aftershocks</div>`;

  sorted.forEach((eq, i) => {
    const card = document.createElement('div');
    card.className = 'eq-card';
    card.dataset.index = i;

    const mag = eq.mainshock_magnitude;
    const as  = eq.predicted_aftershocks ?? [];

    card.innerHTML = `
      <div class="eq-card-top">
        <div class="mag-badge ${magClass(mag)}">M${mag.toFixed(1)}</div>
        <div class="eq-meta">
          <div>${timeAgo(eq.mainshock_timestamp)}</div>
          <div>Depth: ${eq.mainshock_depth_km.toFixed(0)}&thinsp;km</div>
          <div>${eq.distance_to_user_km}&thinsp;km away</div>
        </div>
      </div>
      <div class="eq-location">${eq.mainshock_location || `${fmtCoord(eq.mainshock_latitude,'N','S')}, ${fmtCoord(eq.mainshock_longitude,'E','W')}`}</div>
      <div class="eq-footer">
        <span>${as.length} aftershock${as.length !== 1 ? 's' : ''} predicted</span>
        ${eq.should_alert ? '<span class="alert-pill">&#9888; Alert Zone</span>' : ''}
      </div>
      <div class="expand-hint">&#9660; click to expand aftershocks</div>

      <div class="aftershock-list" id="as-${i}">
        <div class="aftershock-header">Predicted Aftershocks &mdash; ranked by magnitude</div>
        ${as.map(a => `
          <div class="aftershock-row">
            <span class="as-mag">M${a.magnitude.toFixed(1)}</span>
            <span class="as-coord">${fmtCoord(a.latitude,'N','S')}<br>${fmtCoord(a.longitude,'E','W')}</span>
            <div class="conf-bar"><div class="conf-fill" style="width:${(a.confidence * 100).toFixed(0)}%"></div></div>
            <span class="conf-pct">${(a.confidence * 100).toFixed(0)}%</span>
          </div>
        `).join('')}
      </div>
    `;

    card.addEventListener('click', () => {
      const asList = $(`as-${i}`);
      const opening = !asList.classList.contains('open');

      // Close all, remove active
      document.querySelectorAll('.aftershock-list').forEach(l => l.classList.remove('open'));
      document.querySelectorAll('.eq-card').forEach(c => c.classList.remove('active'));
      eqAfterLayer.clearLayers();

      if (opening) {
        asList.classList.add('open');
        card.classList.add('active');
        card.querySelector('.expand-hint').textContent = '▲ click to collapse';
        eqMap.panTo([eq.mainshock_latitude, eq.mainshock_longitude], { animate: true });

        // Draw aftershocks on map
        as.forEach(a => {
          L.circleMarker([a.latitude, a.longitude], {
            radius: Math.max(4, a.magnitude * 2),
            color: '#f97316', fillColor: '#fb923c', fillOpacity: 0.75, weight: 1.5,
          }).addTo(eqAfterLayer)
            .bindPopup(`<b>Predicted Aftershock #${a.rank}</b><br>
              M${a.magnitude.toFixed(1)} &middot; ${(a.confidence * 100).toFixed(0)}% confidence<br>
              ${fmtCoord(a.latitude,'N','S')}, ${fmtCoord(a.longitude,'E','W')}<br>
              Depth: ${a.depth_km.toFixed(0)}&thinsp;km`);
        });
      } else {
        card.querySelector('.expand-hint').textContent = '▼ click to expand aftershocks';
      }
    });

    el.appendChild(card);
  });
}

function plotEqOnMap(list) {
  list.forEach((eq, i) => {
    const mag    = eq.mainshock_magnitude;
    const radius = Math.max(7, mag * 4.5);

    const marker = L.circleMarker([eq.mainshock_latitude, eq.mainshock_longitude], {
      radius,
      color: magColor(mag), fillColor: magColor(mag),
      fillOpacity: 0.55, weight: 2,
    }).addTo(eqMainLayer);

    marker.bindPopup(`
      <b>M${mag.toFixed(1)} &mdash; ${eq.mainshock_location || 'Unknown location'}</b><br>
      ${new Date(eq.mainshock_timestamp).toLocaleString()}<br>
      Depth: ${eq.mainshock_depth_km.toFixed(0)}&thinsp;km<br>
      Distance: ${eq.distance_to_user_km}&thinsp;km from origin<br>
      Predicted aftershocks: ${eq.predicted_aftershocks?.length ?? 0}
    `);

    // Sync with sidebar card
    marker.on('click', () => {
      const card = qs(`.eq-card[data-index="${i}"]`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        card.click();
      }
    });
  });
}

function useMyLocation() {
  if (!navigator.geolocation) { alert('Geolocation not supported in this browser.'); return; }
  navigator.geolocation.getCurrentPosition(
    pos => {
      setEqOrigin(pos.coords.latitude, pos.coords.longitude);
      eqMap.setView([pos.coords.latitude, pos.coords.longitude], 7, { animate: true });
    },
    () => alert('Location access denied or unavailable.'),
  );
}

// ════════════════════════════════════════════════════════════════
//  FLOOD TAB
// ════════════════════════════════════════════════════════════════

let floodMap;
let floodPointMarker = null;
let heatmapLayer, heatmapVisible = false, heatmapFetched = false;
let historicalLayer;
let historicalData = [];
let floodLat = 30.3753, floodLng = 69.3451;

const PK_SW = [23.5, 60.0];
const PK_NE = [37.5, 78.5];

function initFloodMap() {
  floodMap = L.map('flood-map', {
    center: [30.37, 69.34],
    zoom: 5,
    minZoom: 4,
    maxBounds: [[18, 52], [42, 88]],
    maxBoundsViscosity: 0.7,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap &amp; CARTO',
    subdomains: 'abcd', maxZoom: 19,
  }).addTo(floodMap);

  heatmapLayer  = L.layerGroup();
  historicalLayer = L.layerGroup().addTo(floodMap);

  // Pakistan bounding box outline
  L.rectangle([PK_SW, PK_NE], {
    color: '#3b82f6', weight: 1.5, fill: false, dashArray: '5 6', opacity: 0.45,
  }).addTo(floodMap);

  // Default point marker
  floodPointMarker = makeFloodMarker(floodLat, floodLng).addTo(floodMap);

  // Click → check flood at that point
  floodMap.on('click', e => {
    const { lat, lng } = e.latlng;
    if (lat < 23.0 || lat > 38.0 || lng < 59.0 || lng > 79.5) return; // loose bounds
    setFloodPoint(lat, lng);
    queryFlood();
  });

  // Wire up controls
  $('flood-check-btn').addEventListener('click', queryFlood);
  $('heatmap-btn').addEventListener('click', toggleHeatmap);
  $('history-btn').addEventListener('click', () => $('history-overlay').classList.toggle('open'));

  ['flood-lat', 'flood-lng'].forEach(id => {
    $(id).addEventListener('change', () => {
      const lat = parseFloat($('flood-lat').value);
      const lng = parseFloat($('flood-lng').value);
      if (!isNaN(lat) && !isNaN(lng)) setFloodPoint(lat, lng);
    });
  });

  $('flood-date').addEventListener('change', queryFlood);

  // Date setup
  const today = new Date().toISOString().split('T')[0];
  $('flood-date').max   = new Date(Date.now() + 16 * 864e5).toISOString().split('T')[0];
  $('flood-date').value = today;

  loadHistoricalFloods();
}

function makeFloodMarker(lat, lng) {
  return L.marker([lat, lng], {
    icon: L.divIcon({
      html: '<div style="width:14px;height:14px;border-radius:50%;background:#3b82f6;border:2.5px solid #fff;box-shadow:0 0 10px rgba(59,130,246,0.65)"></div>',
      iconSize: [14, 14], iconAnchor: [7, 7], className: '',
    }),
  });
}

function setFloodPoint(lat, lng) {
  floodLat = lat; floodLng = lng;
  $('flood-lat').value = lat.toFixed(4);
  $('flood-lng').value = lng.toFixed(4);
  if (floodPointMarker) floodMap.removeLayer(floodPointMarker);
  floodPointMarker = makeFloodMarker(lat, lng).addTo(floodMap);
}

async function fetchHeatmap() {
  if (heatmapFetched) return;
  heatmapFetched = true;
  try {
    const r = await fetch(`${API_BASE}/flood/heatmap`);
    const d = await r.json();
    if (!d.grid) return;

    d.grid.forEach(pt => {
      const level = pt.risk_level ?? scoreToLevel(pt.risk_score ?? 0);
      L.circleMarker([pt.lat, pt.lon ?? pt.lng], {
        radius: 9, color: 'transparent',
        fillColor: riskColor(level), fillOpacity: 0.38, weight: 0,
      }).addTo(heatmapLayer)
        .bindPopup(`<b>${level}</b><br>Score: ${(pt.risk_score ?? 0).toFixed(1)}/100<br>${fmtCoord(pt.lat,'N','S')}, ${fmtCoord(pt.lon ?? pt.lng,'E','W')}`);
    });
  } catch (e) {
    console.warn('Heatmap fetch failed:', e);
  }
}

function toggleHeatmap() {
  if (!heatmapVisible) {
    if (!heatmapFetched) fetchHeatmap();
    heatmapLayer.addTo(floodMap);
    heatmapVisible = true;
    $('heatmap-btn').textContent = 'Hide Heatmap';
    $('heatmap-btn').classList.add('active-toggle');
  } else {
    floodMap.removeLayer(heatmapLayer);
    heatmapVisible = false;
    $('heatmap-btn').textContent = 'Show Heatmap';
    $('heatmap-btn').classList.remove('active-toggle');
  }
}

async function loadHistoricalFloods() {
  try {
    const r = await fetch(`${API_BASE}/flood/historical`);
    historicalData = await r.json();
    renderHistoryList();
  } catch (e) {
    console.warn('Historical floods fetch failed:', e);
  }
}

function renderHistoryList() {
  $('history-list').innerHTML = historicalData
    .sort((a, b) => b.year - a.year)
    .map(ev => `
      <div class="history-item" onclick="showHistoricalEvent(${ev.year})">
        <div class="hi-year">${ev.year}</div>
        <div class="hi-label">${ev.label}</div>
        <div class="hi-stats">${ev.deaths.toLocaleString()} deaths &middot; ${ev.affected_millions}M affected</div>
      </div>
    `).join('');
}

function showHistoricalEvent(year) {
  historicalLayer.clearLayers();
  const ev = historicalData.find(e => e.year === year);
  if (!ev) return;

  ev.regions.forEach(r => {
    L.circle([r.lat, r.lng], {
      radius: r.radius_km * 1000,
      color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.13, weight: 1.5,
    }).addTo(historicalLayer)
      .bindPopup(`<b>${ev.label} (${ev.year})</b><br>
        ${r.district}<br>
        Deaths: ${ev.deaths.toLocaleString()} &middot; Affected: ${ev.affected_millions}M<br>
        <em style="color:#94a3b8;font-size:11px">${ev.description}</em>`);
  });

  if (historicalLayer.getLayers().length) {
    floodMap.fitBounds(historicalLayer.getBounds().pad(0.08), { animate: true });
  }

  $('history-overlay').classList.remove('open');
}

// Expose for inline onclick
window.showHistoricalEvent = showHistoricalEvent;

async function queryFlood() {
  const date = $('flood-date').value;
  if (!date) return;

  $('flood-result').innerHTML = `<div class="loading"><div class="spinner"></div> Fetching rainfall &amp; running XGBoost model…</div>`;

  try {
    const url = `${API_BASE}/flood/forecast?date=${date}&latitude=${floodLat}&longitude=${floodLng}`;
    const r   = await fetch(url);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail ?? `HTTP ${r.status}`);
    }
    const d = await r.json();
    renderFloodResult(d);
  } catch (e) {
    $('flood-result').innerHTML = `<div class="empty-state"><div class="empty-icon">&#9888;&#65039;</div>${e.message}</div>`;
  }
}

function renderFloodResult(d) {
  const color = riskColor(d.risk_level);
  const score = d.risk_score ?? 0;
  const isPast = d.data_date && d.data_date < new Date().toISOString().split('T')[0];

  $('flood-result').innerHTML = `
    <div class="flood-card">
      <div class="risk-top">
        <div class="risk-label-group">
          <span class="risk-label-sm">${isPast ? 'Historical' : 'Forecast'} Risk</span>
          <span class="risk-badge risk-${d.risk_level}">${d.risk_level}</span>
        </div>
        <div class="risk-score-group">
          <div class="risk-score-val">${score.toFixed(0)}<sub>/100</sub></div>
          <div class="risk-date">${d.data_date ?? ''}</div>
        </div>
      </div>

      <div class="score-bar">
        <div class="score-fill" style="width:${Math.min(score,100)}%; background:${color}"></div>
      </div>

      <div class="flood-stats">
        <div class="stat-box">
          <div class="stat-label">Rainfall</div>
          <div class="stat-val">${d.rainfall_mm.toFixed(1)}<span class="stat-unit"> mm</span></div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Point</div>
          <div class="stat-val" style="font-size:13px;line-height:1.4">
            ${floodLat.toFixed(2)}°N<br>${floodLng.toFixed(2)}°E
          </div>
        </div>
      </div>

      ${d.affected_areas?.length ? `
        <div class="areas-section">
          <div class="areas-title">Nearest Areas</div>
          ${d.affected_areas.map(a => `<span class="area-pill">&#128205; ${a}</span>`).join('')}
        </div>
      ` : ''}

      ${d.should_alert ? `
        <div class="alert-banner">
          &#9888; <strong>High flood risk detected.</strong> Consider checking local emergency services for this area.
        </div>
      ` : ''}
    </div>
  `;
}

// ── Utility ──────────────────────────────────────────────────
function scoreToLevel(s) {
  if (s >= 80) return 'CRITICAL';
  if (s >= 60) return 'HIGH';
  if (s >= 40) return 'MODERATE';
  return 'LOW';
}

// ════════════════════════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
  initEqMap();
  initFloodMap();
  checkHealth();
});
