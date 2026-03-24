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
  if (e.target.classList.contains('chart-link')) {
    const symbol = e.target.dataset.symbol;
    loadChart(symbol, currentPeriod);
  }
});

$('#recTable').addEventListener('dblclick', (e) => {
  const row = e.target.closest('tr');
  if (!row || row.closest('thead')) return;
  const btn = row.querySelector('.chart-link');
  if (btn) loadChart(btn.dataset.symbol, currentPeriod);
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
    renderChart(data.symbol, name, data.candles, smaLabel);
    $('#chartSection').classList.remove('hidden');
    const dispName = name ? `${symbol} — ${name}` : symbol;
    $('#chartTitle').textContent = dispName;
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
async function loadHistory() {
  try {
    const res = await fetch('/api/saved');
    const data = await res.json();
    const list = $('#historyList');

    if (!data.saved.length) {
      list.innerHTML = '<p class="muted">No saved recommendations yet.</p>';
      return;
    }

    list.innerHTML = data.saved
      .map(
        (s) => `
      <div class="history-card" data-filename="${s.filename}">
        <div class="info">
          <strong>${s.filename}</strong>
          <span>${s.count} symbols</span>
          <span>${s.created ? new Date(s.created).toLocaleString() : ''}</span>
        </div>
        <button class="view-btn">View</button>
      </div>
    `
      )
      .join('');
  } catch (err) {
    console.error('Error loading history:', err);
  }
}

$('#historyList').addEventListener('click', async (e) => {
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
});

// --- Init ---
loadHistory();

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
