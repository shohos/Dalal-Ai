import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Message from './models/Message.js';

dotenv.config();
console.log('🔑 GROQ_API_KEY detected:', Boolean(process.env.GROQ_API_KEY));

const app = express();

// Middleware
app.use(cors({ origin: 'http://localhost:5173', credentials: true }));
app.use(express.json());

// DB
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017/mern_chat_mvp';
mongoose
  .connect(MONGODB_URI)
  .then(() => console.log('✅ MongoDB connected'))
  .catch((err) => console.error('MongoDB error:', err));

// Health
app.get('/api/health', (_req, res) => res.json({ ok: true }));

// Debug (safe)
app.get('/api/debug/env', (_req, res) => {
  res.json({
    groq: Boolean(process.env.GROQ_API_KEY),
    model: process.env.GROQ_MODEL || 'llama-3.1-8b-instant'
  });
});

// Conversations: list (coalesce null conversationId to 'default')
app.get('/api/conversations', async (_req, res) => {
  try {
    const rows = await Message.aggregate([
      { $addFields: { cid: { $ifNull: ['$conversationId', 'default'] } } },
      { $sort: { createdAt: -1 } },
      { $group: { _id: '$cid', lastAt: { $first: '$createdAt' }, lastText: { $first: '$text' } } },
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

// Conversations: delete by id
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

// ---- LLM adapter (Groq) ----
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

// Chat: store user & assistant, keep context per conversationId
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

// ---- start server ----
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 API listening on http://localhost:${PORT}`));