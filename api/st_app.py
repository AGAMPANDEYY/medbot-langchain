import os
import time
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

def info():
    ''''
Instructions to run :: Go to api folder cd api and run streamlit run st_app.py 

Limitations::
    - This app only supports English language queries.
    - Their is no conversational RAG chain with Memory (have updated in the notebook version)
    - No evaluation metrics right now
    - Limited PineCone vectors dabase only on few pages of book since takes much time
    - Used Ollama llama model for inefernece that works on my local machine, need to use open source gated models from cloud and need GPUs.
    - Need better datasets
    - Chunking strategy followed is Recurssive Character spliter 
    - Retriver is Cosine similarity

'''

# Load environment variables
load_dotenv()

# Initialize pipeline and session state
pipeline = RAGPipeline()
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not pinecone_api_key:
    st.error("Pinecone API key is missing. Please set it in the environment.")
    st.stop()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page configuration
st.set_page_config(
    page_title="MedBot - Gynecology AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS
st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Message bubbles */
        .user-message {
            background-color: #e0f7fa;  /* Very light cyan */
            color: #004d40;  /* Dark teal text */
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            max-width: 80%;
            margin-left: auto;
        }
        
        .bot-message {
            background-color: #f0f4c3;  /* Light yellow-green */
            color: #33691e;  /* Dark green text */
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            max-width: 80%;
        }

        
        /* Input box styling */
        .stTextInput input {
            border-radius: 25px;
            padding: 10px 20px;
            border: 2px solid #667eea;
            font-size: 16px;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 25px;
            border-radius: 25px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        }
        
        /* Typing indicator */
        .typing-indicator {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:

    # Use os.path.join for a relative path
    image_path=r"C:\Agam\Work\medbot\api\static\media\medbot-logo2.png"
    image=Image.open(image_path)
    st.image(image)
    st.markdown("### About MedBot")
    st.write("""
    MedBot is your AI-powered gynecology assistant. 
    Ask questions about women's health and receive 
    evidence-based responses.
    """)
    
    st.markdown("### Features")
    st.write("✅ 24/7 Availability")
    st.write("✅ Evidence-based Responses")
    st.write("✅ Private & Secure")
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🏥 MedBot - Gynecology AI Assistant")
st.markdown("Your trusted AI companion for gynecological health information")
st.markdown('</div>', unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Create a container for chat messages
chat_container = st.container()

# Create a container for the typing indicator and new messages
response_container = st.container()

# Display chat history in the chat container
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 You: {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🤖 MedBot: {message["content"]}</div>', 
                       unsafe_allow_html=True)

# Input area
question = st.text_input("", placeholder="Type your question here...", key="user_input")
col1, col2 = st.columns([4, 1])

def process_question(question: str):
    if not question.strip():
        return None
    
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Display user message immediately
        with chat_container:
            st.markdown(f'<div class="user-message">👤 You: {question}</div>', 
                       unsafe_allow_html=True)
        
        # Show typing indicator
        with response_container:
            typing_indicator = st.empty()
            typing_indicator.markdown('<div class="typing-indicator">MedBot is typing...</div>', 
                                   unsafe_allow_html=True)
            
            # Get response from pipeline
            answer = pipeline.run_medbot(question)
            
            # Remove typing indicator and show response
            typing_indicator.empty()
            st.markdown(f'<div class="bot-message">🤖 MedBot: {answer}</div>', 
                       unsafe_allow_html=True)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        return answer
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Send button
with col1:
    if st.button("Send", key="send") and question:
        #st.session_state.user_input = ""
        process_question(question)
        # Clear the input box (requires a rerun, but it's optional)

# Voice input button
with col2:
    if st.button("🎤", key="voice"):
        st.info("Voice input feature coming soon!")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made with ❤️ by Your Agam Pandey | © 2024</p>
</div>
""", unsafe_allow_html=True)