// 텔레그램 알림 중계 (브라우저 → 텔레그램 Bot API, CORS 우회용)
// 토큰/챗ID는 사용자의 localStorage에만 저장되며 요청 시에만 전달됨
export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "POST만 지원합니다" });
  }
  const { token, chatId, text } = req.body || {};
  if (!token || !chatId || !text) {
    return res.status(400).json({ error: "token, chatId, text가 필요합니다" });
  }

  try {
    const r = await fetch(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId, text }),
    });
    const json = await r.json();
    return res.status(r.ok ? 200 : 502).json(json);
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
