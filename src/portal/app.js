/* Trading Recommendation Portal — app.js */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let currentRecs = [];
let currentFilters = {};
let chart = null;
let currentSymbol = null;
let currentPeriod = '1m';
let currentChartType = 'candle';
let lastChartData = null;

// --- Generate ---
$('#generateBtn').addEventListener('click', generate);

async function generate() {
  const btn = $('#generateBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-spinner"></span>Generating...';
  showStatus('loading', 'Analyzing 421 symbols...');

  const params = new URLSearchParams({
    num: $('#numSymbols').value,
    profit: $('#profitTarget').value,
    hold: $('#holdDays').value,
    sma: $('#smaFilter').value,
    vol: $('#volFilter').value,
  });

  const cutoff = $('#cutoffDate').value;
  if (cutoff) params.set('cutoff', cutoff + ' 19:30:00');

  currentFilters = Object.fromEntries(params.entries());

  try {
    const res = await fetch('/api/recommend?' + params);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    currentRecs = data.recommendations;
    renderTable(currentRecs);
    $('#results').classList.remove('hidden');
    $('#savedBanner').classList.add('hidden');
    $('#saveBtn').disabled = false;
    showStatus('success', `Found ${currentRecs.length} recommendations`);
    setTimeout(() => $('#status').classList.add('hidden'), 3000);
  } catch (err) {
    showStatus('error', 'Error: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

function showStatus(type, msg) {
  const el = $('#status');
  el.className = type;
  if (type === 'loading') {
    el.innerHTML = '<span class="spinner"></span>' + msg;
  } else {
    el.textContent = msg;
  }
}

// --- Symbol names ---
const SYMBOL_NAMES = {}; // populated from API responses

// --- Table ---
function renderTable(recs) {
  const tbody = $('#recTable tbody');
  tbody.innerHTML = '';
  recs.forEach((r, i) => {
    const name = r.name || SYMBOL_NAMES[r.symbol] || '';
    if (r.name) SYMBOL_NAMES[r.symbol] = r.name;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td><strong>${r.symbol}</strong>${name ? '<span class="sym-name">' + name + '</span>' : ''}</td>
      <td>$${r.current_price.toFixed(2)}</td>
      <td>${r.score}</td>
      <td>$${r.target_price.toFixed(2)}</td>
      <td>$${r.stop_loss.toFixed(2)}</td>
      <td class="${r.expected_profit >= 0 ? 'positive' : 'negative'}">${r.expected_profit.toFixed(1)}%</td>
      <td class="${r.ytd_pct >= 0 ? 'positive' : 'negative'}">${r.ytd_pct.toFixed(1)}%</td>
      <td class="${r.month_pct >= 0 ? 'positive' : 'negative'}">${r.month_pct.toFixed(1)}%</td>
      <td>${r.vol_label}</td>
      <td><button class="info-btn" data-symbol="${r.symbol}" title="Symbol Info">ℹ</button></td>
      <td><button class="chart-link" data-symbol="${r.symbol}">View</button></td>
    `;
    tbody.appendChild(tr);
  });
}

// --- Chart ---
$('#recTable').addEventListener('click', (e) => {
  // Always handle View button click
  if (e.target.classList.contains('chart-link')) {
    const symbol = e.target.dataset.symbol;
    loadChart(symbol, currentPeriod);
    return;
  }
  // On mobile portrait: single click anywhere on a data row opens fullscreen chart
  if (isMobilePortrait()) {
    const row = e.target.closest('tr');
    if (!row || row.closest('thead')) return;
    // Skip info button
    if (e.target.classList.contains('info-btn') || e.target.closest('.info-btn')) return;
    const btn = row.querySelector('.chart-link');
    if (btn) loadChart(btn.dataset.symbol, currentPeriod);
  }
});

$('#recTable').addEventListener('dblclick', (e) => {
  const row = e.target.closest('tr');
  if (!row || row.closest('thead')) return;
  const btn = row.querySelector('.chart-link');
  if (btn) loadChart(btn.dataset.symbol, currentPeriod);
});

// --- Mobile fullscreen chart ---
function isMobilePortrait() {
  return window.innerWidth <= 768;
}

function openMobileChart() {
  if (!isMobilePortrait()) return;
  hideMobileInfoPanel();
  document.body.classList.add('mobile-chart-open');
  history.pushState({ mobileChart: true }, '');
  updateMobileNavCounter();
  // Resize chart to fit fullscreen
  setTimeout(() => {
    if (chart) {
      const container = $('#priceChart');
      chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
      chart.timeScale().fitContent();
    }
  }, 50);
}

function closeMobileChart() {
  document.body.classList.remove('mobile-chart-open');
  // Hide chart section so it doesn't show below the list on mobile
  if (isMobilePortrait()) {
    $('#chartSection').classList.add('hidden');
  }
  // Resize chart back
  setTimeout(() => {
    if (chart) {
      const container = $('#priceChart');
      chart.applyOptions({ width: container.clientWidth, height: 450 });
    }
  }, 50);
}

$('#chartBackBtn').addEventListener('click', () => {
  closeMobileChart();
  if (history.state && history.state.mobileChart) history.back();
});

// --- Mobile swipe gestures ---
(function () {
  let sx = 0, sy = 0;
  const el = $('#chartSection');

  el.addEventListener('touchstart', (e) => {
    sx = e.touches[0].clientX;
    sy = e.touches[0].clientY;
  }, { passive: true });

  el.addEventListener('touchend', (e) => {
    if (!isMobilePortrait() || !document.body.classList.contains('mobile-chart-open')) return;
    const dx = e.changedTouches[0].clientX - sx;
    const dy = e.changedTouches[0].clientY - sy;
    const adx = Math.abs(dx), ady = Math.abs(dy);

    if (adx >= 55 && adx > ady * 1.4) {
      // Horizontal: navigate symbols
      if (dx < 0) navigateSymbol(+1); // swipe left → next
      else         navigateSymbol(-1); // swipe right → prev
    } else if (ady >= 65 && ady > adx * 1.4) {
      if (dy < 0) showMobileInfoPanel(); // swipe up → symbol info slides down
      else        navigateSymbol(+1);    // swipe down → next symbol
    }
  }, { passive: true });
})();

// Prev / Next buttons
$('#mobilePrevBtn').addEventListener('click', () => navigateSymbol(-1));
$('#mobileNextBtn').addEventListener('click', () => navigateSymbol(+1));

// Tap handle to dismiss info panel
$('#mobileInfoPanel').addEventListener('click', (e) => {
  if (e.target.classList.contains('mobile-info-handle')) hideMobileInfoPanel();
});

function navigateSymbol(dir) {
  if (!currentRecs.length) return;
  const idx = currentRecs.findIndex((r) => r.symbol === currentSymbol);
  const next = idx + dir;
  if (next < 0 || next >= currentRecs.length) return;
  loadChart(currentRecs[next].symbol, currentPeriod);
}

function updateMobileNavCounter() {
  const counter = $('#mobileNavCounter');
  const prevBtn = $('#mobilePrevBtn');
  const nextBtn = $('#mobileNextBtn');
  if (!counter) return;
  const idx = currentRecs.findIndex((r) => r.symbol === currentSymbol);
  if (idx >= 0 && currentRecs.length > 0) {
    counter.textContent = `${idx + 1} / ${currentRecs.length}`;
    if (prevBtn) prevBtn.disabled = idx === 0;
    if (nextBtn) nextBtn.disabled = idx === currentRecs.length - 1;
  } else {
    counter.textContent = '';
    if (prevBtn) prevBtn.disabled = true;
    if (nextBtn) nextBtn.disabled = true;
  }
}

async function showMobileInfoPanel() {
  const panel = $('#mobileInfoPanel');
  const content = $('#mobileInfoContent');
  if (!panel || !currentSymbol) return;
  panel.classList.add('info-open');
  // Only re-fetch if symbol changed
  if (panel.dataset.loadedSymbol === currentSymbol) return;
  panel.dataset.loadedSymbol = currentSymbol;
  content.innerHTML = '<p class="muted" style="text-align:center;padding:2rem 0"><span class="spinner"></span> Loading info...</p>';
  try {
    const res = await fetch(`/api/info/${encodeURIComponent(currentSymbol)}`);
    if (!res.ok) throw new Error('Not available');
    const data = await res.json();
    content.innerHTML = renderInfoContent(data);
    if (data.profile && data.profile.name) {
      // keep the handle at top
    }
  } catch {
    content.innerHTML = `<p class="muted" style="text-align:center;padding:2rem 0">No info available for ${currentSymbol}</p>`;
  }
}

function hideMobileInfoPanel() {
  const panel = $('#mobileInfoPanel');
  if (panel) panel.classList.remove('info-open');
}

window.addEventListener('popstate', (e) => {
  if (document.body.classList.contains('mobile-chart-open')) {
    closeMobileChart();
  }
});

$$('.period-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.period-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    currentPeriod = btn.dataset.period;
    if (currentSymbol) loadChart(currentSymbol, currentPeriod);
  });
});

$$('.type-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    $$('.type-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    currentChartType = btn.dataset.type;
    if (lastChartData) {
      renderChart(lastChartData.symbol, lastChartData.name, lastChartData.candles, lastChartData.smaLabel);
    }
  });
});

async function loadChart(symbol, period) {
  currentSymbol = symbol;
  const params = new URLSearchParams({ period });
  const cutoff = $('#cutoffDate').value;
  if (cutoff) params.set('cutoff', cutoff + ' 19:30:00');
  const sma = $('#smaFilter').value;
  const smaVal = sma && sma !== '0' ? sma : '20';
  params.set('sma', smaVal);

  try {
    const res = await fetch(`/api/chart/${encodeURIComponent(symbol)}?` + params);
    if (!res.ok) throw new Error('Chart data not found');
    const data = await res.json();
    const name = data.name || SYMBOL_NAMES[symbol] || '';
    if (data.name) SYMBOL_NAMES[symbol] = data.name;
    const smaLabel = `SMA-${smaVal}`;
    lastChartData = { symbol: data.symbol, name, candles: data.candles, smaLabel };
    // Make chart section visible BEFORE rendering so container has dimensions
    $('#chartSection').classList.remove('hidden');
    const dispName = name ? `${symbol} — ${name}` : symbol;
    $('#chartTitle').textContent = dispName;
    // Open fullscreen chart on mobile (also resets info panel + updates counter)
    openMobileChart();
    updateMobileNavCounter();
    // Small delay to let the DOM layout update, then render
    requestAnimationFrame(() => {
      renderChart(data.symbol, name, data.candles, smaLabel);
    });
    // Fetch live quote
    fetchLiveQuote(symbol);
    startQuoteRefresh(symbol);
  } catch (err) {
    console.error(err);
  }
}

function renderChart(symbol, name, candles, smaLabel) {
  const container = $('#priceChart');

  // Destroy previous chart
  if (chart) {
    chart.remove();
    chart = null;
  }
  container.innerHTML = '';

  const isCandlestick = currentChartType === 'candle';

  chart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 450,
    layout: {
      background: { color: '#12141c' },
      textColor: '#999',
    },
    grid: {
      vertLines: { color: '#1e2030' },
      horzLines: { color: '#1e2030' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2d3a' },
    timeScale: {
      borderColor: '#2a2d3a',
      timeVisible: true,
      secondsVisible: false,
    },
  });

  // Convert candle times to UTC timestamps (seconds)
  const toTimestamp = (t) => Math.floor(new Date(t).getTime() / 1000);

  if (isCandlestick) {
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#66bb6a',
      downColor: '#ef5350',
      borderUpColor: '#66bb6a',
      borderDownColor: '#ef5350',
      wickUpColor: '#66bb6a',
      wickDownColor: '#ef5350',
    });
    candleSeries.setData(
      candles.map((c) => ({
        time: toTimestamp(c.time),
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );
  } else {
    const lineSeries = chart.addLineSeries({
      color: '#4fc3f7',
      lineWidth: 2,
    });
    lineSeries.setData(
      candles.map((c) => ({
        time: toTimestamp(c.time),
        value: c.close,
      }))
    );
  }

  // SMA line
  const smaPoints = candles
    .filter((c) => c.sma != null)
    .map((c) => ({ time: toTimestamp(c.time), value: c.sma }));
  if (smaPoints.length > 0) {
    const smaSeries = chart.addLineSeries({
      color: '#ffb74d',
      lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      title: smaLabel || 'SMA',
    });
    smaSeries.setData(smaPoints);
  }

  // Volume histogram
  const volSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'vol',
  });
  chart.priceScale('vol').applyOptions({
    scaleMargins: { top: 0.8, bottom: 0 },
  });
  volSeries.setData(
    candles.map((c) => ({
      time: toTimestamp(c.time),
      value: c.volume,
      color: c.close >= c.open ? 'rgba(102, 187, 106, 0.5)' : 'rgba(239, 83, 80, 0.5)',
    }))
  );

  // Resize on window resize
  const resizeHandler = () => {
    if (chart) chart.applyOptions({ width: container.clientWidth });
  };
  window.removeEventListener('resize', window._chartResizeHandler);
  window._chartResizeHandler = resizeHandler;
  window.addEventListener('resize', resizeHandler);

  chart.timeScale().fitContent();
}

// --- Live Quote ---
let quoteInterval = null;

async function fetchLiveQuote(symbol) {
  const el = $('#liveQuote');
  el.textContent = 'Loading live price...';
  el.className = 'live-quote';
  try {
    const res = await fetch(`/api/quote/${encodeURIComponent(symbol)}`);
    if (!res.ok) { el.textContent = ''; return; }
    const q = await res.json();
    const sign = q.change >= 0 ? '+' : '';
    const cls = q.change >= 0 ? 'positive' : 'negative';
    el.innerHTML = `<span class="live-price">$${q.price.toFixed(2)}</span> <span class="${cls}">${sign}${q.change.toFixed(2)} (${sign}${q.change_pct.toFixed(2)}%)</span> <span class="live-label">Live</span>`;
  } catch {
    el.textContent = '';
  }
}

function startQuoteRefresh(symbol) {
  if (quoteInterval) clearInterval(quoteInterval);
  quoteInterval = setInterval(() => fetchLiveQuote(symbol), 30000);
}

function stopQuoteRefresh() {
  if (quoteInterval) { clearInterval(quoteInterval); quoteInterval = null; }
}

// --- Save ---
$('#saveBtn').addEventListener('click', async () => {
  if (!currentRecs.length) return;

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recommendations: currentRecs,
        filters: currentFilters,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showStatus('success', `Saved as ${data.filename}`);
    setTimeout(() => $('#status').classList.add('hidden'), 3000);
    loadHistory();
  } catch (err) {
    showStatus('error', 'Save failed: ' + err.message);
  }
});

// --- History ---
let historySortAsc = true;
let historySavedData = [];

async function loadHistory() {
  try {
    const res = await fetch('/api/saved');
    const data = await res.json();
    historySavedData = data.saved || [];
    renderHistory();
  } catch (err) {
    console.error('Error loading history:', err);
  }
}

function renderHistory() {
  const list = $('#historyList');
  const sorted = [...historySavedData].sort((a, b) => {
    const cmp = a.filename.localeCompare(b.filename);
    return historySortAsc ? cmp : -cmp;
  });

  if (!sorted.length) {
    const emptyMsg = '<p class="muted">No saved recommendations yet.</p>';
    list.innerHTML = emptyMsg;
    syncHistoryLists();
    return;
  }

  list.innerHTML = sorted
    .map(
      (s) => `
    <div class="history-card" data-filename="${s.filename}">
      <div class="info">
        <strong>${s.filename}</strong>
        <span>${s.count} symbols</span>
        <span>${s.created ? new Date(s.created).toLocaleString() : ''}</span>
      </div>
      <div class="history-actions">
        <button class="view-btn">View</button>
        <button class="delete-btn" data-filename="${s.filename}" title="Delete">&#x1f5d1;</button>
      </div>
    </div>
  `
    )
    .join('');
  syncHistoryLists();
  updateSortButtons();
}

function toggleHistorySort() {
  historySortAsc = !historySortAsc;
  renderHistory();
}

function updateSortButtons() {
  const arrow = historySortAsc ? '\u25B2' : '\u25BC';
  const title = historySortAsc ? 'Sort Z→A' : 'Sort A→Z';
  $$('.sort-btn').forEach(btn => { btn.innerHTML = arrow; btn.title = title; });
}

$('#sortBtnSidebar').addEventListener('click', toggleHistorySort);
$('#sortBtnDefault').addEventListener('click', toggleHistorySort);

async function handleHistoryClick(e) {
  // Handle delete button
  if (e.target.classList.contains('delete-btn')) {
    e.stopPropagation();
    const filename = e.target.dataset.filename;
    if (!confirm(`Delete ${filename}?`)) return;
    try {
      const res = await fetch(`/api/saved/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Delete failed');
      showStatus('success', `Deleted ${filename}`);
      setTimeout(() => $('#status').classList.add('hidden'), 3000);
      loadHistory();
    } catch (err) {
      showStatus('error', 'Delete failed: ' + err.message);
    }
    return;
  }

  const card = e.target.closest('.history-card');
  if (!card) return;
  const filename = card.dataset.filename;

  try {
    const res = await fetch(`/api/saved/${encodeURIComponent(filename)}`);
    if (!res.ok) throw new Error('File not found');
    const data = await res.json();
    currentRecs = data.recommendations;
    currentFilters = data.filters || {};
    renderTable(currentRecs);
    $('#results').classList.remove('hidden');
    removeLandscapePlaceholder();

    // Highlight active card in sidebar
    setActiveHistoryCard(filename);

    // Show saved file banner with filters
    const f = currentFilters;
    const filterText = [
      f.num && `${f.num} symbols`,
      f.hold && `${f.hold}d hold`,
      f.profit && `${f.profit}% profit`,
      f.sma && f.sma !== '0' && `SMA-${f.sma}`,
      f.vol && f.vol !== 'None' && `Vol: ${f.vol}`,
    ].filter(Boolean).join(' | ');
    const banner = $('#savedBanner');
    banner.innerHTML = `<strong>${filename}</strong><span class="banner-filters">${filterText}</span>`;
    banner.classList.remove('hidden');
    $('#saveBtn').disabled = true;

    showStatus('success', `Loaded ${filename}`);
    setTimeout(() => $('#status').classList.add('hidden'), 3000);
  } catch (err) {
    showStatus('error', 'Error loading: ' + err.message);
  }
}

$('#historyList').addEventListener('click', handleHistoryClick);
$('#historyListDefault').addEventListener('click', handleHistoryClick);

// --- Landscape split panel ---
let _landscapeBusy = false;

function isLandscape() {
  return window.innerWidth > 768 && window.matchMedia('(orientation: landscape)').matches;
}

function enableLandscapeMode() {
  if (_landscapeBusy) return;
  _landscapeBusy = true;

  try {
    const lc = $('#landscapeContent');
    const main = document.querySelector('main');
    const wrap = $('#landscapeWrap');

    if (!isLandscape()) {
      if (!document.body.classList.contains('landscape-mode')) { _landscapeBusy = false; return; }
      document.body.classList.remove('landscape-mode');
      // Move sections back to main in correct order
      const filters = lc.querySelector('#filters');
      const status = lc.querySelector('#status');
      const results = lc.querySelector('#results');
      const chartSection = lc.querySelector('#chartSection');
      if (filters) main.insertBefore(filters, wrap);
      if (status) main.insertBefore(status, wrap);
      if (results) main.insertBefore(results, wrap);
      if (chartSection) main.insertBefore(chartSection, wrap);
      _landscapeBusy = false;
      return;
    }

    document.body.classList.add('landscape-mode');

    // Move filters, status, results + chart into landscape content panel (in order)
    const filters = $('#filters');
    const status = $('#status');
    const results = $('#results');
    const chartSection = $('#chartSection');

    // Clear placeholder first
    removeLandscapePlaceholder();

    // Insert in correct order: filters first
    if (filters && filters.parentElement !== lc) lc.prepend(filters);
    if (status && status.parentElement !== lc) {
      const after = lc.querySelector('#filters');
      if (after) after.after(status); else lc.prepend(status);
    }
    if (results && results.parentElement !== lc) lc.appendChild(results);
    if (chartSection && chartSection.parentElement !== lc) lc.appendChild(chartSection);

    // Show placeholder if nothing loaded yet
    if (results.classList.contains('hidden') && !lc.querySelector('.landscape-placeholder')) {
      const ph = document.createElement('div');
      ph.className = 'landscape-placeholder';
      ph.textContent = 'Select a saved recommendation or generate new ones';
      lc.appendChild(ph);
    }
  } finally {
    _landscapeBusy = false;
  }
}

function removeLandscapePlaceholder() {
  const ph = document.querySelector('.landscape-placeholder');
  if (ph) ph.remove();
}

// Sync sidebar history list with default history list
function syncHistoryLists() {
  const defaultList = $('#historyListDefault');
  const sidebarList = $('#historyList');
  if (defaultList && sidebarList) {
    defaultList.innerHTML = sidebarList.innerHTML;
  }
}

let _resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(enableLandscapeMode, 80);
});
window.matchMedia('(orientation: landscape)').addEventListener('change', () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(enableLandscapeMode, 80);
});

// Mark active card in sidebar
function setActiveHistoryCard(filename) {
  $$('#historySidebar .history-card').forEach(c => c.classList.remove('active'));
  const card = document.querySelector(`#historySidebar .history-card[data-filename="${filename}"]`);
  if (card) card.classList.add('active');
}

// --- Init ---
loadHistory();
setTimeout(enableLandscapeMode, 100);

// --- Symbol Info Modal ---
$('#recTable').addEventListener('click', (e) => {
  if (e.target.classList.contains('info-btn')) {
    e.stopPropagation();
    openInfoModal(e.target.dataset.symbol);
  }
});

$('#infoModalClose').addEventListener('click', () => {
  $('#infoModal').classList.add('hidden');
});

$('#infoModal').addEventListener('click', (e) => {
  if (e.target === $('#infoModal')) {
    $('#infoModal').classList.add('hidden');
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#infoModal').classList.contains('hidden')) {
    $('#infoModal').classList.add('hidden');
  }
});

async function openInfoModal(symbol) {
  const modal = $('#infoModal');
  const body = $('#infoModalBody');
  const title = $('#infoModalTitle');
  const name = SYMBOL_NAMES[symbol] || '';
  title.textContent = name ? `${symbol} — ${name}` : symbol;
  body.innerHTML = '<p class="muted" style="text-align:center;padding:2rem 0;"><span class="spinner"></span> Loading info...</p>';
  modal.classList.remove('hidden');

  try {
    const res = await fetch(`/api/info/${encodeURIComponent(symbol)}`);
    if (!res.ok) throw new Error('Info not available');
    const data = await res.json();
    body.innerHTML = renderInfoContent(data);
    // Update title with full name from API
    if (data.profile && data.profile.name) {
      title.textContent = `${symbol} — ${data.profile.name}`;
    }
  } catch (err) {
    body.innerHTML = `<p class="muted" style="text-align:center;padding:2rem 0;">Could not load info for ${symbol}</p>`;
  }
}

function renderInfoContent(data) {
  const fmt = (v, type) => {
    if (v === null || v === undefined) return '—';
    if (type === 'dollar') return '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (type === 'pct') return (Number(v) * 100).toFixed(2) + '%';
    if (type === 'big') {
      const n = Number(v);
      if (n >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
      if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
      if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
      return '$' + n.toLocaleString();
    }
    if (type === 'num') return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 });
    if (type === 'int') return Number(v).toLocaleString();
    return String(v);
  };

  const section = (title, rows) => {
    const validRows = rows.filter(([, v]) => v !== null && v !== undefined && v !== '—');
    if (!validRows.length) return '';
    return `<div class="info-section"><h3>${title}</h3><div class="info-grid">${
      validRows.map(([label, value]) => `<div class="info-row"><span class="info-label">${label}</span><span class="info-value">${value}</span></div>`).join('')
    }</div></div>`;
  };

  const p = data.profile || {};
  const pr = data.price || {};
  const m = data.market || {};
  const v = data.valuation || {};
  const f = data.financials || {};
  const d = data.dividends || {};
  const a = data.analyst || {};

  let html = '';

  // Profile description
  if (p.description) {
    html += `<div class="info-section"><h3>About</h3><div class="info-description">${p.description}</div></div>`;
  }

  html += section('Company', [
    ['Sector', p.sector],
    ['Industry', p.industry],
    ['Country', p.country],
    ['Employees', fmt(p.employees, 'int')],
    ['Website', p.website ? `<a href="${p.website}" target="_blank" rel="noopener" style="color:#4fc3f7">${p.website}</a>` : null],
  ]);

  html += section('Price', [
    ['Current', fmt(pr.current, 'dollar')],
    ['Open', fmt(pr.open, 'dollar')],
    ['Prev Close', fmt(pr.previous_close, 'dollar')],
    ['Day Low', fmt(pr.day_low, 'dollar')],
    ['Day High', fmt(pr.day_high, 'dollar')],
    ['52W Low', fmt(pr['52w_low'], 'dollar')],
    ['52W High', fmt(pr['52w_high'], 'dollar')],
    ['50D Avg', fmt(pr['50d_avg'], 'dollar')],
    ['200D Avg', fmt(pr['200d_avg'], 'dollar')],
  ]);

  html += section('Market', [
    ['Market Cap', fmt(m.market_cap, 'big')],
    ['Enterprise Value', fmt(m.enterprise_value, 'big')],
    ['Volume', fmt(m.volume, 'int')],
    ['Avg Volume', fmt(m.avg_volume, 'int')],
    ['Shares Outstanding', fmt(m.shares_outstanding, 'int')],
    ['Float', fmt(m.float_shares, 'int')],
    ['Beta', fmt(m.beta, 'num')],
  ]);

  html += section('Valuation', [
    ['P/E (TTM)', fmt(v.pe_trailing, 'num')],
    ['P/E (Fwd)', fmt(v.pe_forward, 'num')],
    ['PEG', fmt(v.peg_ratio, 'num')],
    ['P/B', fmt(v.price_to_book, 'num')],
    ['P/S', fmt(v.price_to_sales, 'num')],
    ['EV/EBITDA', fmt(v.ev_to_ebitda, 'num')],
    ['EV/Revenue', fmt(v.ev_to_revenue, 'num')],
  ]);

  html += section('Financials', [
    ['Revenue', fmt(f.revenue, 'big')],
    ['Gross Profit', fmt(f.gross_profit, 'big')],
    ['EBITDA', fmt(f.ebitda, 'big')],
    ['Net Income', fmt(f.net_income, 'big')],
    ['EPS (TTM)', fmt(f.eps_trailing, 'num')],
    ['EPS (Fwd)', fmt(f.eps_forward, 'num')],
    ['Profit Margin', fmt(f.profit_margin, 'pct')],
    ['Operating Margin', fmt(f.operating_margin, 'pct')],
    ['Gross Margin', fmt(f.gross_margin, 'pct')],
    ['ROE', fmt(f.return_on_equity, 'pct')],
    ['ROA', fmt(f.return_on_assets, 'pct')],
    ['Debt/Equity', fmt(f.debt_to_equity, 'num')],
    ['Current Ratio', fmt(f.current_ratio, 'num')],
    ['Free Cash Flow', fmt(f.free_cash_flow, 'big')],
    ['Operating Cash Flow', fmt(f.operating_cash_flow, 'big')],
  ]);

  html += section('Dividends', [
    ['Dividend Rate', fmt(d.dividend_rate, 'dollar')],
    ['Dividend Yield', fmt(d.dividend_yield, 'pct')],
    ['Payout Ratio', fmt(d.payout_ratio, 'pct')],
  ]);

  html += section('Analyst Targets', [
    ['Recommendation', a.recommendation ? a.recommendation.toUpperCase() : null],
    ['# Analysts', fmt(a.num_analysts, 'int')],
    ['Target Low', fmt(a.target_low, 'dollar')],
    ['Target Median', fmt(a.target_median, 'dollar')],
    ['Target Mean', fmt(a.target_mean, 'dollar')],
    ['Target High', fmt(a.target_high, 'dollar')],
  ]);

  return html || '<p class="muted">No data available.</p>';
}
