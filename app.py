import streamlit as st
from langchain.memory import ConversationBufferMemory
from mldl_chatboat import SmartSearch   # Import your SmartSearch class
import time

st.set_page_config(
    page_title="📚 MindMiner ",
    page_icon="🤖",
    layout="wide"
)

# 💾 Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🌟 Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("**AI PDF + Web Assistant**")
st.sidebar.info("Search your PDFs + Web seamlessly with memory context.")
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2)

st.sidebar.markdown("---")
st.sidebar.write("💡 **Tip**: Ask follow-up questions without repeating context!")

# 🎯 Main Header
st.markdown(
    "<h1 style='text-align:center;'>📚 AI PDF + Web Assistant</h1>",
    unsafe_allow_html=True
)

# 💬 Chat Input
user_input = st.chat_input("Ask me something...")

if user_input:
    # Show user message instantly
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Typing animation
    with st.spinner("🤔 Thinking..."):
        time.sleep(0.2)  # Small delay for effect
        search_tool = SmartSearch(user_input, st.session_state.memory)
        answer = search_tool.search()

    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 🖼 Display Chat
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"🧑 **You:** {chat['content']}")
    else:
        st.markdown(f"🤖 **Assistant:** {chat['content']}")

# 📌 Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Built with ❤️ using LangChain, Streamlit & Gemini</div>",
    unsafe_allow_html=True

)
