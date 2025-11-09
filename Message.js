import mongoose from 'mongoose';

const MessageSchema = new mongoose.Schema(
  {
    conversationId: { type: String, index: true, default: 'default' },
    role: { type: String, enum: ['user', 'assistant'], required: true },
    text: { type: String, required: true }
  },
  { timestamps: true }
);

export default mongoose.model('Message', MessageSchema);