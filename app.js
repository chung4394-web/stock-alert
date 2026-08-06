/* ==========================================================
 * 주식 AI 상승하락 알리미
 *  - Yahoo Finance 데이터(2년 일봉)로 기술적 지표 계산
 *  - 로지스틱 회귀 모델을 브라우저에서 직접 학습
 *  - 핵심 판단 기준: 수급(거래량 흐름) — OBV, MFI, 상승/하락일 거래량
 *    "상승은 수급이 따라올 때 붙는다" → 수급 점수가 최종 판단을 좌우
 * ========================================================== */

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "stockAlert.watchlist";

let watchlist = JSON.parse(localStorage.getItem(STORAGE_KEY) || '["005930.KS"]');
let watchTimer = null;
let notifiedToday = {}; // { "심볼|날짜|방향": true } 중복 알림 방지

/* ─────────── 데이터 ─────────── */

async function fetchChart(symbol, range, interval) {
  const r = await fetch(
    `/api/stock?symbol=${encodeURIComponent(symbol)}&range=${range}&interval=${interval}`
  );
  if (!r.ok) throw new Error(`API 오류 (${r.status})`);
  const json = await r.json();
  const result = json?.chart?.result?.[0];
  if (!result) throw new Error(json?.chart?.error?.description || "데이터 없음");
  return result;
}

/* ─────────── 기술적 지표 ─────────── */

function sma(arr, n, i) {
  if (i < n - 1) return null;
  let s = 0;
  for (let k = i - n + 1; k <= i; k++) s += arr[k];
  return s / n;
}

function stdev(arr, n, i) {
  const m = sma(arr, n, i);
  if (m === null) return null;
  let s = 0;
  for (let k = i - n + 1; k <= i; k++) s += (arr[k] - m) ** 2;
  return Math.sqrt(s / n);
}

function computeRSI(close) {
  const period = 14;
  const rsi = new Array(close.length).fill(null);
  let gain = 0, loss = 0;
  for (let i = 1; i < close.length; i++) {
    const d = close[i] - close[i - 1];
    if (i <= period) {
      if (d > 0) gain += d; else loss -= d;
      if (i === period) {
        gain /= period; loss /= period;
        rsi[i] = 100 - 100 / (1 + gain / (loss || 1e-9));
      }
    } else {
      gain = (gain * (period - 1) + Math.max(d, 0)) / period;
      loss = (loss * (period - 1) + Math.max(-d, 0)) / period;
      rsi[i] = 100 - 100 / (1 + gain / (loss || 1e-9));
    }
  }
  return rsi;
}

function ema(arr, span) {
  const k = 2 / (span + 1);
  const out = new Array(arr.length);
  out[0] = arr[0];
  for (let i = 1; i < arr.length; i++) out[i] = arr[i] * k + out[i - 1] * (1 - k);
  return out;
}

/* ─────────── 수급 지표 ─────────── */

// OBV(On-Balance Volume): 상승일엔 거래량을 더하고 하락일엔 빼는 누적값.
// 가격보다 먼저 움직이는 "매집/분산"의 흔적을 보여준다.
function computeOBV(close, volume) {
  const obv = new Array(close.length).fill(0);
  for (let i = 1; i < close.length; i++) {
    if (close[i] > close[i - 1]) obv[i] = obv[i - 1] + volume[i];
    else if (close[i] < close[i - 1]) obv[i] = obv[i - 1] - volume[i];
    else obv[i] = obv[i - 1];
  }
  return obv;
}

// MFI(Money Flow Index): 거래량을 반영한 RSI. 50 이상이면 자금 유입 우위.
function computeMFI(high, low, close, volume) {
  const period = 14;
  const mfi = new Array(close.length).fill(null);
  const tp = close.map((c, i) => (high[i] + low[i] + c) / 3);
  for (let i = period; i < close.length; i++) {
    let pos = 0, neg = 0;
    for (let k = i - period + 1; k <= i; k++) {
      const mf = tp[k] * volume[k];
      if (tp[k] > tp[k - 1]) pos += mf;
      else if (tp[k] < tp[k - 1]) neg += mf;
    }
    mfi[i] = 100 - 100 / (1 + pos / (neg || 1e-9));
  }
  return mfi;
}

// 최근 20일 중 상승일 거래량 합 / 하락일 거래량 합
function upDownVolRatio(close, volume, i, n = 20) {
  let up = 0, down = 0;
  for (let k = i - n + 1; k <= i; k++) {
    if (close[k] > close[k - 1]) up += volume[k];
    else if (close[k] < close[k - 1]) down += volume[k];
  }
  return (up + 1) / (down + 1);
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));

// 수급 점수 (0~100): OBV 추세 40% + MFI 30% + 상승/하락 거래량비 30%
function supplyScore(obvSlope, mfi, udRatio) {
  const obvComp = clamp01(0.5 + obvSlope / 0.3);
  const mfiComp = clamp01(mfi / 100);
  const udComp = clamp01(0.5 + Math.log(udRatio) / (2 * Math.log(4)));
  return {
    score: Math.round((0.4 * obvComp + 0.3 * mfiComp + 0.3 * udComp) * 100),
    obvComp, mfiComp, udComp,
  };
}

/* ─────────── 데이터셋 생성 ─────────── */

// 각 날짜별 특징 벡터(기술 지표 + 수급 지표) + 라벨(다음날 상승=1)
function buildDataset(close, volume, high, low) {
  const n = close.length;
  const ret1 = close.map((c, i) => (i > 0 ? c / close[i - 1] - 1 : null));
  const rsi = computeRSI(close);
  const ema12 = ema(close, 12);
  const ema26 = ema(close, 26);
  const macd = ema12.map((v, i) => v - ema26[i]);
  const signal = ema(macd, 9);
  const obv = computeOBV(close, volume);
  const mfi = computeMFI(high, low, close, volume);

  const rows = [];
  for (let i = 60; i < n; i++) {
    const ma5 = sma(close, 5, i), ma20 = sma(close, 20, i), ma60 = sma(close, 60, i);
    const sd20 = stdev(close, 20, i);
    const volMa20 = sma(volume, 20, i);
    // 20일 수익률 변동성
    let vsum = 0, vcnt = 0, vmean = 0;
    for (let k = i - 19; k <= i; k++) vmean += ret1[k];
    vmean /= 20;
    for (let k = i - 19; k <= i; k++) { vsum += (ret1[k] - vmean) ** 2; vcnt++; }
    const volatility = Math.sqrt(vsum / vcnt);

    // 수급 지표
    const obvSlope = volMa20 > 0 ? (obv[i] - obv[i - 20]) / (volMa20 * 20) : 0;
    const udRatio = upDownVolRatio(close, volume, i);

    const feat = [
      ret1[i],
      close[i] / close[i - 5] - 1,
      close[i] / close[i - 20] - 1,
      close[i] / ma5 - 1,
      close[i] / ma20 - 1,
      close[i] / ma60 - 1,
      (rsi[i] - 50) / 50,
      (macd[i] - signal[i]) / close[i],
      sd20 > 0 ? (close[i] - (ma20 - 2 * sd20)) / (4 * sd20) - 0.5 : 0,
      volatility,
      volMa20 > 0 ? volume[i] / volMa20 - 1 : 0,
      obvSlope,
      ((mfi[i] ?? 50) - 50) / 50,
      Math.log(udRatio),
    ];
    if (feat.some((v) => v === null || !isFinite(v))) continue;
    const label = i + 1 < n ? (close[i + 1] > close[i] ? 1 : 0) : null;
    rows.push({ feat, label, idx: i, obvSlope, mfi: mfi[i] ?? 50, udRatio });
  }
  return rows;
}

/* ─────────── 로지스틱 회귀 (브라우저 학습) ─────────── */

function standardize(rows, stats) {
  const dim = rows[0].feat.length;
  if (!stats) {
    stats = { mean: new Array(dim).fill(0), sd: new Array(dim).fill(0) };
    for (const r of rows) r.feat.forEach((v, j) => (stats.mean[j] += v));
    stats.mean = stats.mean.map((m) => m / rows.length);
    for (const r of rows) r.feat.forEach((v, j) => (stats.sd[j] += (v - stats.mean[j]) ** 2));
    stats.sd = stats.sd.map((s) => Math.sqrt(s / rows.length) || 1);
  }
  const X = rows.map((r) => r.feat.map((v, j) => (v - stats.mean[j]) / stats.sd[j]));
  return { X, stats };
}

function trainLogistic(X, y, epochs = 400, lr = 0.05, l2 = 0.001) {
  const dim = X[0].length;
  let w = new Array(dim).fill(0), b = 0;
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  for (let e = 0; e < epochs; e++) {
    const gw = new Array(dim).fill(0);
    let gb = 0;
    for (let i = 0; i < X.length; i++) {
      let z = b;
      for (let j = 0; j < dim; j++) z += w[j] * X[i][j];
      const err = sigmoid(z) - y[i];
      for (let j = 0; j < dim; j++) gw[j] += err * X[i][j];
      gb += err;
    }
    for (let j = 0; j < dim; j++) w[j] -= lr * (gw[j] / X.length + l2 * w[j]);
    b -= lr * (gb / X.length);
  }
  return {
    predict(x) {
      let z = b;
      for (let j = 0; j < x.length; j++) z += w[j] * x[j];
      return sigmoid(z);
    },
  };
}

/* ─────────── 종합 판단 ───────────
 * 원칙: "상승은 수급이 따라올 때 붙는다"
 *  - 수급 점수 ≥ 60 + 모델 상승 → 상승 (수급 뒷받침)
 *  - 수급 점수 ≥ 60 + 모델 하락 → 반등 주시 (수급 유입 중)
 *  - 수급 점수 ≤ 40             → 하락 우려 (수급 이탈) — 모델과 무관
 *  - 그 외                      → 모델 방향 따르되 수급 중립 표시
 */
function makeVerdict(probUp, supply) {
  const modelUp = probUp >= 0.5;
  if (supply.score >= 60 && modelUp)
    return { dir: "up", text: "📈 상승", reason: "수급 유입 + 모델 상승 신호" };
  if (supply.score >= 60)
    return { dir: "up", text: "👀 반등 주시", reason: "수급 유입 중, 모델은 하락 신호" };
  if (supply.score <= 40)
    return { dir: "down", text: "📉 하락 우려", reason: "수급 이탈 — 상승 신호 신뢰 낮음" };
  return modelUp
    ? { dir: "up", text: "🤏 약한 상승", reason: "모델 상승 신호, 수급 중립" }
    : { dir: "down", text: "📉 하락", reason: "모델 하락 신호, 수급 중립" };
}

// 전체 파이프라인: 학습 → 백테스트 → 최신 데이터 예측 + 수급 판단
function analyze(close, volume, high, low) {
  const rows = buildDataset(close, volume, high, low);
  const labeled = rows.filter((r) => r.label !== null);
  if (labeled.length < 150) throw new Error("데이터가 부족합니다");

  const split = Math.floor(labeled.length * 0.8);
  const trainRows = labeled.slice(0, split);
  const testRows = labeled.slice(split);

  const { X: Xtrain, stats } = standardize(trainRows);
  const model = trainLogistic(Xtrain, trainRows.map((r) => r.label));

  const { X: Xtest } = standardize(testRows, stats);
  let correct = 0, ups = 0;
  testRows.forEach((r, i) => {
    if ((model.predict(Xtest[i]) >= 0.5 ? 1 : 0) === r.label) correct++;
    ups += r.label;
  });
  const accuracy = correct / testRows.length;
  const baseline = Math.max(ups, testRows.length - ups) / testRows.length;

  const latest = rows[rows.length - 1];
  const { X: Xlatest } = standardize([latest], stats);
  const probUp = model.predict(Xlatest[0]);

  const supply = supplyScore(latest.obvSlope, latest.mfi, latest.udRatio);
  const verdict = makeVerdict(probUp, supply);

  return { probUp, accuracy, baseline, supply, verdict };
}

/* ─────────── UI: 예측 카드 ─────────── */

function fmtPrice(p, currency) {
  const digits = p >= 1000 ? 0 : 2;
  return p.toLocaleString("ko-KR", { maximumFractionDigits: digits }) +
    (currency === "KRW" ? "원" : currency === "USD" ? " $" : ` ${currency || ""}`);
}

function supplyLabel(score) {
  if (score >= 60) return "💧 수급 유입";
  if (score <= 40) return "🏃 수급 이탈";
  return "➖ 수급 중립";
}

async function renderCard(symbol) {
  const card = document.createElement("div");
  card.className = "card loading";
  card.id = `card-${symbol}`;
  card.innerHTML = `
    <button class="remove" title="삭제">✕</button>
    <h3>${symbol}</h3>
    <div class="symbol">불러오는 중...</div>
    <div class="pred"><div class="meta">AI 모델 학습 중...</div></div>`;
  card.querySelector(".remove").onclick = () => removeSymbol(symbol);
  $("cards").appendChild(card);

  try {
    const result = await fetchChart(symbol, "2y", "1d");
    const meta = result.meta;
    const quote = result.indicators.quote[0];
    const close = [], volume = [], high = [], low = [];
    quote.close.forEach((c, i) => {
      if (c !== null && quote.volume[i] !== null &&
          quote.high[i] !== null && quote.low[i] !== null) {
        close.push(c);
        volume.push(quote.volume[i]);
        high.push(quote.high[i]);
        low.push(quote.low[i]);
      }
    });

    const { probUp, accuracy, supply, verdict } = analyze(close, volume, high, low);
    const price = meta.regularMarketPrice ?? close[close.length - 1];
    // chartPreviousClose는 차트 범위(2년) 시작 전 종가라서 쓰면 안 됨 — 전일 종가 사용
    const prevClose = meta.regularMarketPreviousClose ?? close[close.length - 2];
    const changePct = ((price - prevClose) / prevClose) * 100;

    card.className = "card";
    card.innerHTML = `
      <button class="remove" title="삭제">✕</button>
      <h3>${meta.shortName || symbol}</h3>
      <div class="symbol">${symbol}</div>
      <div>
        <span class="price">${fmtPrice(price, meta.currency)}</span>
        <span class="change ${changePct >= 0 ? "up" : "down"}">
          ${changePct >= 0 ? "▲" : "▼"} ${Math.abs(changePct).toFixed(2)}%
        </span>
      </div>
      <div class="supply">
        <div class="supply-head">
          <span>${supplyLabel(supply.score)}</span>
          <strong>${supply.score}점</strong>
        </div>
        <div class="prob-bar supply-bar"><div style="width:${supply.score}%"></div></div>
        <div class="meta">OBV ${supply.obvComp >= 0.5 ? "매집↑" : "분산↓"} ·
          MFI ${Math.round(supply.mfiComp * 100)} ·
          상승일 거래량 ${supply.udComp >= 0.5 ? "우위" : "열세"}</div>
      </div>
      <div class="pred">
        <div class="dir ${verdict.dir}">${verdict.text}</div>
        <div class="meta reason">${verdict.reason}</div>
        <div class="prob-bar"><div style="width:${(probUp * 100).toFixed(0)}%"></div></div>
        <div class="meta">모델 상승확률 ${(probUp * 100).toFixed(0)}% · 정확도 ${(accuracy * 100).toFixed(0)}%</div>
      </div>`;
    card.querySelector(".remove").onclick = () => removeSymbol(symbol);
  } catch (e) {
    card.className = "card";
    card.querySelector(".symbol").textContent = `⚠️ ${e.message}`;
    card.querySelector(".pred").innerHTML =
      `<div class="meta">종목코드를 확인하세요 (한국: 005930.KS, 미국: AAPL)</div>`;
  }
}

function renderAll() {
  $("cards").innerHTML = "";
  watchlist.forEach(renderCard);
}

function addSymbol() {
  const s = $("symbol-input").value.trim().toUpperCase();
  if (!s) return;
  if (watchlist.includes(s)) { alert("이미 등록된 종목입니다"); return; }
  watchlist.push(s);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist));
  $("symbol-input").value = "";
  renderCard(s);
}

function removeSymbol(symbol) {
  watchlist = watchlist.filter((s) => s !== symbol);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(watchlist));
  $(`card-${symbol}`)?.remove();
}

/* ─────────── 알림 ─────────── */

function addLog(msg) {
  const empty = $("log").querySelector(".empty");
  if (empty) empty.remove();
  const li = document.createElement("li");
  const t = new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
  li.innerHTML = `<time>${t}</time>${msg}`;
  $("log").prepend(li);
}

function notify(title, body) {
  addLog(`<strong>${title}</strong> — ${body}`);
  if (Notification.permission === "granted") {
    new Notification(title, { body });
  }
}

async function checkPrices() {
  const threshold = parseFloat($("threshold").value);
  const today = new Date().toISOString().slice(0, 10);
  for (const symbol of watchlist) {
    try {
      const result = await fetchChart(symbol, "1d", "5m");
      const meta = result.meta;
      const price = meta.regularMarketPrice;
      const prev = meta.regularMarketPreviousClose ?? meta.chartPreviousClose;
      if (!price || !prev) continue;
      const changePct = ((price - prev) / prev) * 100;
      const dir = changePct >= 0 ? "up" : "down";
      const key = `${symbol}|${today}|${dir}`;
      if (Math.abs(changePct) >= threshold && !notifiedToday[key]) {
        notifiedToday[key] = true;
        const name = meta.shortName || symbol;
        notify(
          `${name} ${changePct >= 0 ? "📈" : "📉"} ${changePct >= 0 ? "+" : ""}${changePct.toFixed(1)}%`,
          fmtPrice(price, meta.currency)
        );
      }
    } catch { /* 개별 종목 오류는 무시하고 계속 */ }
  }
  $("watch-status").textContent =
    `마지막 확인: ${new Date().toLocaleTimeString("ko-KR")} · ${watchlist.length}개 종목 감시 중`;
}

function startWatching() {
  if (watchTimer) clearInterval(watchTimer);
  const minutes = parseInt($("check-interval").value);
  watchTimer = setInterval(checkPrices, minutes * 60 * 1000);
  checkPrices();
}

async function enableNotifications() {
  if (!("Notification" in window)) {
    alert("이 브라우저는 알림을 지원하지 않습니다");
    return;
  }
  const perm = await Notification.requestPermission();
  if (perm === "granted") {
    $("notify-btn").textContent = "🔔 알림 켜짐";
    $("notify-btn").classList.add("on");
    notify("알림 활성화 완료", "설정한 변동률을 넘으면 알려드립니다");
    startWatching();
  } else {
    alert("브라우저 알림 권한이 거부되었습니다. 브라우저 설정에서 허용해주세요.");
  }
}

/* ─────────── 초기화 ─────────── */

$("add-btn").onclick = addSymbol;
$("symbol-input").addEventListener("keydown", (e) => { if (e.key === "Enter") addSymbol(); });
$("notify-btn").onclick = enableNotifications;
$("check-now-btn").onclick = checkPrices;
$("check-interval").addEventListener("change", () => { if (watchTimer) startWatching(); });

$("log").innerHTML = '<li class="empty">아직 알림이 없습니다</li>';
if ("Notification" in window && Notification.permission === "granted") {
  $("notify-btn").textContent = "🔔 알림 켜짐";
  $("notify-btn").classList.add("on");
  startWatching();
}
renderAll();
