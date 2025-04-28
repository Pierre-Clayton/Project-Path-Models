# utils/helpers.py
import streamlit as st
from datetime import datetime

def initialize_session_state():
    """Initializes all required session state variables."""
    defaults = {
        "left_messages": [], "right_messages": [],
        "left_model": "gpt-4o", "right_model": "gpt-4o-mini",
        "left_mbti": "INTJ", "right_mbti": "INTJ",
        "left_big_five": {"openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50, "neuroticism": 50},
        "right_big_five": {"openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50, "neuroticism": 50},
        "use_big_five": False,
        "OPENAI_API_KEY": "", # Loaded from env/secrets preferred
        "jira_url": "", "jira_email": "", "jira_api_token": "",
        "jira_connected": False, "jira_client": None,
        "jira_projects": [], "selected_jira_project_keys": [],
        "project_context": "", "use_project_context": True,
        # --- RAG State ---
        "uploaded_files_info": [], # Store names/sources of indexed files
        "vector_store": None, # Holds the FAISS index
        "rag_index_ready": False,
        "use_rag_context": True,
        "processing_docs": False, # Flag to prevent multiple indexing runs
        # --- Drive Connector State ---
        "google_creds": None,
        "google_service": None,
        "google_files": None, # List of {'id': file_id, 'name': file_name}
        "ms_token": None, # MSAL token cache
        "ms_account": None,
        "ms_files": None, # List of {'id': file_id, 'name': file_name}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_personality_profile(mbti=None, big_five=None, use_mbti=True, use_big_five=False):
    """Creates a string representation of the personality profile."""
    profile = []
    if use_mbti and mbti:
        profile.append(f"MBTI Type: {mbti}")
    if use_big_five and big_five:
        profile.append("Big Five Personality Traits:")
        for trait, value in big_five.items():
            level = "low" if value < 33 else "moderate" if value < 66 else "high"
            profile.append(f"- {trait.capitalize()}: {level} ({value}%)")
    return "\n".join(profile)

def format_message(role, content, model_name=None, timestamp=None):
    """Formats a message dictionary."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    msg = {"role": role, "content": content, "timestamp": timestamp}
    if model_name:
        msg["model_name"] = model_name # Store which model generated it
    return msg

def display_chat_message(message):
    """Displays a single chat message using Streamlit markdown."""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    model_name = message.get("model_name", "Assistant") # Get model name if available

    if role == "user":
        st.markdown(f"<div class='user-message'><b>You:</b> {content}<div class='message-timestamp'>{timestamp}</div></div>", unsafe_allow_html=True)
    elif role == "assistant":
        # Use the stored model name for display
        st.markdown(f"<div class='assistant-message'><b>{model_name}:</b> {content}<div class='message-timestamp'>{timestamp}</div></div>", unsafe_allow_html=True)
    elif role == "system": # Optional: display system messages/context for debugging
        st.markdown(f"<div style='color: grey; font-size: 0.8em;'><i>System: {content}</i></div>", unsafe_allow_html=True)