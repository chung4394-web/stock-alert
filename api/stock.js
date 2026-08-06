// Yahoo Finance 차트 API 프록시 (브라우저 CORS 우회용)
export default async function handler(req, res) {
  const { symbol, range = "2y", interval = "1d" } = req.query;
  if (!symbol) {
    return res.status(400).json({ error: "symbol 파라미터가 필요합니다" });
  }

  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}` +
    `?range=${encodeURIComponent(range)}&interval=${encodeURIComponent(interval)}`;

  try {
    const r = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0 (stock-alert app)" },
    });
    if (!r.ok) {
      return res.status(r.status).json({ error: `Yahoo API 오류 (${r.status})` });
    }
    const json = await r.json();
    res.setHeader("Cache-Control", "s-maxage=60, stale-while-revalidate=300");
    return res.status(200).json(json);
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
