// backend/index.js
import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Message from './models/Message.js';

dotenv.config();

console.log('🔑 GROQ_API_KEY detected:', Boolean(process.env.GROQ_API_KEY));
const app = express();

// Basic middleware
app.use(cors({ origin: 'http://localhost:5173', credentials: true }));
app.use(express.json({ limit: '1mb' }));

// Connect DB
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/mern_chat_mvp';
mongoose
  .connect(MONGODB_URI)
  .then(() => console.log('✅ MongoDB connected'))
  .catch((err) => console.error('MongoDB error:', err));

// Health
app.get('/api/health', (_req, res) => res.json({ ok: true }));

// Debug (does not reveal secrets)
app.get('/api/debug/env', (_req, res) => {
  res.json({
    groq: Boolean(process.env.GROQ_API_KEY),
    model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
    predictUrl: Boolean(process.env.PREDICT_URL)
  });
});

// ---------- Conversations ----------
app.get('/api/conversations', async (_req, res) => {
  try {
    const rows = await Message.aggregate([
      { $sort: { createdAt: -1 } },
      { $group: { _id: '$conversationId', lastAt: { $first: '$createdAt' }, lastText: { $first: '$text' } } },
      { $project: { conversationId: '$_id', lastAt: 1, lastText: 1, _id: 0 } },
      { $sort: { lastAt: -1 } },
      { $limit: 50 }
    ]);
    res.json(rows);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to list conversations' });
  }
});

app.delete('/api/conversations/:cid', async (req, res) => {
  try {
    const cid = req.params.cid.toString();
    await Message.deleteMany({ conversationId: cid });
    res.json({ ok: true });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Failed to delete conversation' });
  }
});

// Fetch chat history
app.get('/api/messages', async (req, res) => {
  try {
    const cid = (req.query.cid || 'default').toString();
    const msgs = await Message.find({ conversationId: cid }).sort({ createdAt: 1 }).limit(200);
    res.json(msgs);
  } catch (e) {
    res.status(500).json({ error: 'Failed to fetch messages' });
  }
});

// ---------- LLM adapter (Groq) ----------
async function getAssistantReply(prompt, history = [], lang = 'bn') {
  const key = process.env.GROQ_API_KEY;
  const systemPrompt =
    lang === 'bn'
      ? 'তুমি একটি বাংলা-ভাষী সহকারী। ব্যবহারকারীর প্রশ্ন যদি ইংরেজিতে হয়, তখনই ইংরেজিতে উত্তর দাও; অন্যথায় সব উত্তর বাংলায় দেবে। সংক্ষিপ্ত, সহায়ক এবং সংখ্যাগুলো ভুল কোরো না।'
      : 'You are a helpful assistant. If the user writes in Bangla, answer in Bangla; otherwise reply in English. Be concise and numerically accurate.';

  if (!key) {
    if (/price|best|deal|range/i.test(prompt)) {
      return 'এখানে একটি ছোট টিপস: সাম্প্রতিক মিল-সদৃশ লিস্টিং দেখো এবং মিডিয়ান দামের চেয়ে ১০–১৫% কম দিয়ে শুরু করো। মডেল প্লাগইন হলে আমি স্বয়ংক্রিয়ভাবে কম্পস দেখাব।';
    }
    return `তুমি লিখেছ: "${prompt}"। LLM মোড চালু নয়; GROQ_API_KEY দিলে বাস্তব উত্তর আসবে।`;
  }

  const messages = [
    { role: 'system', content: systemPrompt },
    ...history.map((m) => ({ role: m.role, content: m.text })),
    { role: 'user', content: prompt }
  ];

  try {
    const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`
      },
      body: JSON.stringify({
        model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
        messages,
        temperature: 0.2
      })
    });

    if (!resp.ok) {
      const errText = await resp.text();
      console.error('Groq error:', errText);
      return 'LLM-এ যোগাযোগে সমস্যা হয়েছে। আপাতত fallback টিপস দেখাচ্ছি।';
    }
    const data = await resp.json();
    return data.choices?.[0]?.message?.content?.trim() || '…';
  } catch (err) {
    console.error(err);
    return 'LLM রিকোয়েস্ট ব্যর্থ হয়েছে।';
  }
}

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { text, conversationId, lang = 'bn' } = req.body || {};
    const cid = (conversationId || 'default').toString();
    if (!text || !text.trim()) return res.status(400).json({ error: 'Text is required' });

    const history = await Message.find({ conversationId: cid }).sort({ createdAt: 1 }).limit(40);

    const userMsg = await Message.create({ conversationId: cid, role: 'user', text: text.trim() });
    const reply = await getAssistantReply(text.trim(), history, lang);
    const botMsg = await Message.create({ conversationId: cid, role: 'assistant', text: reply });

    res.json({ messages: [userMsg, botMsg] });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Something went wrong' });
  }
});

// ---------- Model proxy (→ FastAPI /predict) ----------
const REQUIRED_FIELDS = [
  'category', 'brand_tier', 'condition', 'season', 'division', 'delivery_zone',
  'seller_rating', 'stock', 'shipping_days', 'demand_index',
  'competitor_price_bdt', 'cost_bdt', 'discount_pct',
  'clicks_last_7d', 'views_last_7d', 'conversions_last_7d', 'time_on_market_days',
  'bkash_share', 'nagad_share', 'cod_share', 'card_share',
  'is_weekend', 'is_ramadan', 'is_eid', 'is_puja', 'is_boishakh', 'vat_included'
];

function coerceToItems(body) {
  if (!body) return [];
  if (Array.isArray(body.items)) return body.items;
  if (body.item && typeof body.item === 'object') return [body.item];
  // support sending a single flat object
  if (typeof body === 'object' && !Array.isArray(body)) return [body];
  return [];
}

function validateItem(item) {
  const missing = REQUIRED_FIELDS.filter((k) => !(k in item));
  return { ok: missing.length === 0, missing };
}

app.get('/api/model/health', async (_req, res) => {
  try {
    const url = process.env.PREDICT_URL || 'http://127.0.0.1:8001/predict';
    // Tiny ping with a minimal valid row
    const sample = {
      category: 'laptop', brand_tier: 'mid', condition: 'new', season: 'winter',
      division: 'Dhaka', delivery_zone: 'Dhaka-Metro',
      seller_rating: 4.5, stock: 50, shipping_days: 2, demand_index: 0.6,
      competitor_price_bdt: 90000, cost_bdt: 60000, discount_pct: 0.05,
      clicks_last_7d: 100, views_last_7d: 300, conversions_last_7d: 10, time_on_market_days: 7,
      bkash_share: 0.4, nagad_share: 0.2, cod_share: 0.3, card_share: 0.1,
      is_weekend: 0, is_ramadan: 0, is_eid: 0, is_puja: 0, is_boishakh: 0, vat_included: 1
    };
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items: [sample] })
    });
    const j = await r.json().catch(() => ({}));
    res.json({ ok: r.ok, status: r.status, body: j });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok: false, error: 'Model health check failed' });
  }
});

app.post('/api/predict', async (req, res) => {
  try {
    const url = process.env.PREDICT_URL || 'http://127.0.0.1:8001/predict';
    const items = coerceToItems(req.body);

    if (!items.length) return res.status(400).json({ error: 'No items provided' });
    for (let i = 0; i < items.length; i++) {
      const { ok, missing } = validateItem(items[i]);
      if (!ok) return res.status(400).json({ error: `Missing fields in item[${i}]: ${missing.join(', ')}` });
    }

    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items })
    });

    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }

    if (!r.ok) {
      console.error('Model error:', data);
      return res.status(r.status).json({ error: 'Model service error', detail: data });
    }

    res.json(data); // { predictions: [...] }
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Prediction failed' });
  }
});

// ---------- Boot ----------
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 API listening on http://localhost:${PORT}`));
