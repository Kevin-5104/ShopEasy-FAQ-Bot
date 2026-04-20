# ============================================================
# capstone_streamlit.py  —  ShopEasy FAQ Bot UI
# Run with:  streamlit run capstone_streamlit.py
# ============================================================

import streamlit as st
import uuid
from agent import build_app, ask

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ShopEasy Support Bot",
    page_icon="🛒",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f4fd 100%);
    }
    .chat-bubble-user {
        background: #0066ff;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0;
        max-width: 80%;
        float: right;
        clear: both;
        font-size: 14px;
    }
    .chat-bubble-bot {
        background: white;
        color: #1a1a2e;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 0;
        max-width: 80%;
        float: left;
        clear: both;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 3px solid #0066ff;
    }
    .meta-tag {
        font-size: 11px;
        color: #888;
        clear: both;
        padding: 2px 8px;
    }
    .header-box {
        background: linear-gradient(135deg, #0066ff, #0044cc);
        color: white;
        padding: 20px 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header-box h1 { margin: 0; font-size: 26px; font-weight: 700; }
    .header-box p  { margin: 4px 0 0; opacity: 0.85; font-size: 13px; }
    .sidebar-section {
        background: white;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 2px solid #0066ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load app (cached so it only loads once) ──────────────────
@st.cache_resource
def load_bot():
    return build_app()

bot = load_bot()

# ── Session state ────────────────────────────────────────────
if "messages"  not in st.session_state:
    st.session_state.messages  = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🛒 ShopEasy Support Bot")
    st.markdown("**Domain:** E-Commerce FAQ")
    st.markdown("**Built with:** LangGraph + ChromaDB + Groq")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📚 Topics I can help with:**")
    topics = [
        "🔄 Return Policy", "💰 Refund Process", "🚚 Shipping & Delivery",
        "❌ Order Cancellation", "💳 Payment Methods", "🔐 Account & Login",
        "🛡️ Product Warranty", "🎁 Offers & Coupons",
        "📞 Customer Support", "✅ Product Authenticity"
    ]
    for t in topics:
        st.markdown(f"- {t}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**📞 Helpline:** 1800-123-4567")
    st.markdown("**⏰ Support Hours:** 8 AM – 10 PM IST")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.user_name = ""
        st.rerun()

    st.markdown(f"<small>Session ID: `{st.session_state.thread_id[:8]}...`</small>",
                unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛒 ShopEasy Support</h1>
    <p>Your 24/7 shopping assistant — ask me anything about orders, returns, payments & more!</p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ─────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True)
        if msg.get("meta"):
            st.markdown(f'<div class="meta-tag">{msg["meta"]}</div>', unsafe_allow_html=True)
    st.markdown('<div style="clear:both"></div>', unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────
user_input = st.chat_input("Type your question here... (e.g. How do I return a product?)")

if user_input and user_input.strip():
    # Add user message to display
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("ShopBot is thinking..."):
        # Prepend name context if we know it and it's not in this message
        question_to_send = user_input
        if st.session_state.user_name and "my name is" not in user_input.lower():
            question_to_send = f"[Customer name: {st.session_state.user_name}] {user_input}"
        result = ask(bot, question_to_send, st.session_state.thread_id)

    if result.get("user_name"):
        st.session_state.user_name = result["user_name"]

    meta = f"Route: {result['route']} | Faithfulness: {result['faithfulness']:.2f}"
    if result["sources"]:
        meta += f" | Sources: {', '.join(result['sources'])}"

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "meta":    meta
    })
    st.rerun()

# ── Empty state ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; color:#888; padding:40px 0;">
        <p style="font-size:40px">👋</p>
        <p style="font-size:16px">Hi! I'm ShopBot, your ShopEasy assistant.</p>
        <p style="font-size:13px">Ask me about returns, refunds, shipping, payments, and more!</p>
    </div>
    """, unsafe_allow_html=True)
