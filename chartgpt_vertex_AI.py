import os
import streamlit as st
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel
from streamlit_mic_recorder import mic_recorder

# Initialize Vertex AI
vertexai.init(
    project=os.environ["GCP_PROJECT_ID"],
    location=os.environ["GCP_LOCATION"]
)

model = GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="ChartGPT Using Vertex AI", page_icon="🤖", layout="wide")

# --- Sidebar Chat History ---
st.sidebar.title("💬 Chat History")
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, msg in enumerate(st.session_state.messages):
    st.sidebar.write(f"{i+1}. **{msg['role'].capitalize()}**: {msg['content'][:30]}...")

# --- Main Title ---
st.title("🤖 My ChartGPT Using Vertex AI")

# --- Image Upload ---
uploaded_image = st.file_uploader("📷 Upload an image (optional)", type=["jpg", "jpeg", "png"])
if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", width=250)

# --- Text + Voice Input ---
col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.text_input("Ask me anything (or use the mic):", "")

with col2:
    audio = mic_recorder(
        start_prompt="🎤", 
        stop_prompt="🛑",
        just_once=True, 
        use_container_width=True
    )

if audio and not user_input:
    st.success("Voice recorded! Converting to text is optional. You can use this for future STT integration.")

# --- Send Button ---
if st.button("Send") and (user_input or uploaded_image):
    query = user_input if user_input else "User sent an image."

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate AI response
    if uploaded_image:
        response = model.generate_content(
            [query, Image.open(uploaded_image)],
            generation_config={"max_output_tokens": 200}
        )
    else:
        response = model.generate_content(
            query,
            generation_config={"max_output_tokens": 200}
        )

    ai_text = response.text
    st.session_state.messages.append({"role": "ai", "content": ai_text})

    # Display conversation
    st.subheader("💬 Chat Conversation:")
    for msg in st.session_state.messages:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")



