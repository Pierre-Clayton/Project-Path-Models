# app.py
import streamlit as st
import openai
import os
import logging
from datetime import datetime

# --- Local Imports ---
# Ensure these paths are correct relative to app.py
from utils.config import (
    JIRA_TOKEN_URL, AVAILABLE_MODELS, AVAILABLE_FINE_TUNED_MODELS, ALL_MODELS,
    FINE_TUNED_PROMPTS, DEFAULT_SPECIAL_FT_PROMPT, PERSONAS, MBTI_TYPES,
    SUPPORTED_DOC_TYPES, VECTOR_STORE_DIR
)
from utils.helpers import (
    initialize_session_state, create_personality_profile,
    format_message, display_chat_message
)
from utils.jira_utils import (
    connect_jira, get_jira_projects, format_jira_data
)
from utils.rag_utils import (
    load_document, split_documents, create_vector_store,
    get_retriever, format_retrieved_docs
)
from utils.drive_connectors import (
    get_google_drive_service, list_google_drive_files, download_google_drive_file,
    get_ms_graph_client, acquire_ms_token_interactive, list_onedrive_files, download_onedrive_file
)

# --- Basic Config ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(
    page_title="ProjectPathLM Model Comparison (Jira & RAG)",
    page_icon="🤖",
    layout="wide"
)

# --- Initialize Session State ---
# This function should define all keys used in the app
initialize_session_state()

# --- Load Secrets & Set API Key ---
# Try loading OpenAI Key from secrets first
if "OPENAI_API_KEY" not in st.session_state or not st.session_state.OPENAI_API_KEY:
    try:
        st.session_state.OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        logging.info("Loaded OpenAI API Key from secrets.")
    except KeyError:
        st.session_state.OPENAI_API_KEY = ""
        logging.info("OpenAI API Key not found in secrets, waiting for user input.")
    except Exception as e:
        st.warning(f"Could not read Streamlit secrets: {e}. API keys must be entered manually.")
        st.session_state.OPENAI_API_KEY = ""

# Ensure os environment variable is set if key exists in session state
if st.session_state.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = st.session_state.OPENAI_API_KEY
    openai.api_key = st.session_state.OPENAI_API_KEY


# --- Custom CSS ---
st.markdown("""
<style>
    /* Add or keep your existing CSS */
    .chat-container { border-radius: 10px; padding: 15px; background-color: #f9f9f9; margin-bottom: 10px; max-height: 600px; overflow-y: auto; }
    .user-message { background-color: #E3F2FD; border-radius: 10px; padding: 10px; margin: 5px 0; color: #000000; }
    .assistant-message { background-color: #F5F5F5; border-radius: 10px; padding: 10px; margin: 5px 0; color: #000000; }
    .message-timestamp { font-size: 0.7em; color: #616161; text-align: right; }
    .stButton>button { width: 100%; }
    .title-container { text-align: center; margin-bottom: 20px; }
    .personality-section { background-color: #8e7f7c; padding: 10px; border-radius: 5px; margin-top: 10px; color: white; }
    .sidebar .stButton>button { margin-top: 5px; margin-bottom: 5px; }
    .small-font { font-size: 0.9em; color: #666; }
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='title-container'><h1>ProjectPath Model Comparison (Jira & RAG Context)</h1></div>", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # OpenAI Key Input (keep this as a fallback/override)
    api_key_input = st.text_input(
        "OpenAI API Key:",
        type="password",
        value=st.session_state.OPENAI_API_KEY,
        key="openai_api_key_input_sidebar", # Use distinct key
        help="Enter key or set via .streamlit/secrets.toml (OPENAI_API_KEY)."
    )
    if api_key_input and api_key_input != st.session_state.OPENAI_API_KEY:
        st.session_state.OPENAI_API_KEY = api_key_input
        os.environ["OPENAI_API_KEY"] = api_key_input
        openai.api_key = api_key_input
        st.success("OpenAI API Key updated.")
        # Rerun might be needed if embeddings depend on the key at init
        # Let user decide or trigger based on specific action.

    st.divider()

    # --- Jira Configuration ---
    with st.expander("Connect to Jira", expanded=not st.session_state.get("jira_connected", False)):
        st.text_input("Jira Instance URL", key="jira_url", placeholder="your-domain.atlassian.net", value=st.session_state.get("jira_url", ""))
        st.caption("E.g., `your-company.atlassian.net`")
        st.text_input("Jira Email", key="jira_email", placeholder="your-email@example.com", value=st.session_state.get("jira_email", ""))
        st.text_input("Jira API Token", key="jira_api_token", type="password", help=f"Generate token: {JIRA_TOKEN_URL}", value=st.session_state.get("jira_api_token", ""))

        col_con, col_discon = st.columns(2)
        with col_con:
            if st.button("Connect Jira", key="connect_jira_btn", disabled=st.session_state.get("jira_connected", False)):
                if not st.session_state.jira_url or not st.session_state.jira_email or not st.session_state.jira_api_token:
                     st.error("Please provide Jira URL, Email, and API Token.")
                else:
                    with st.spinner("Connecting to Jira..."):
                        client, error_msg = connect_jira(
                            st.session_state.jira_url,
                            st.session_state.jira_email,
                            st.session_state.jira_api_token
                        )
                        if client:
                            st.session_state.jira_client = client
                            st.session_state.jira_connected = True
                            st.session_state.jira_projects = get_jira_projects(client)
                            st.success("Jira connected!")
                            st.rerun()
                        else:
                            st.error(f"Jira Connection Failed: {error_msg}")
                            st.session_state.jira_connected = False
                            st.session_state.jira_client = None
        with col_discon:
             if st.button("Disconnect Jira", key="disconnect_jira_btn", disabled=not st.session_state.get("jira_connected", False)):
                st.session_state.jira_client = None
                st.session_state.jira_connected = False
                st.session_state.jira_projects = []
                st.session_state.selected_jira_project_keys = []
                st.session_state.project_context = ""
                st.success("Jira disconnected.")
                st.rerun()

        if st.session_state.get("jira_connected", False):
            st.success(f"Connected to: {st.session_state.jira_url}")

    # --- Jira Context Selection ---
    if st.session_state.get("jira_connected", False):
        with st.expander("Jira Project Context", expanded=True):
            st.session_state.use_project_context = st.toggle(
                "Use Jira Context",
                value=st.session_state.get("use_project_context", True),
                key="toggle_jira_context"
                )

            jira_project_options = {proj.key: f"{proj.name} ({proj.key})" for proj in st.session_state.get("jira_projects", [])}

            if not jira_project_options:
                 st.warning("No Jira projects found or accessible.")
            else:
                default_selection = st.session_state.get("selected_jira_project_keys", [])
                if not isinstance(default_selection, list): default_selection = []

                st.session_state.selected_jira_project_keys = st.multiselect(
                    "Select Jira Projects:",
                    options=list(jira_project_options.keys()),
                    format_func=lambda proj_key: jira_project_options.get(proj_key, proj_key),
                    default=default_selection,
                    key="jira_project_multiselect"
                )

                if st.button("Fetch/Update Jira Data", key="fetch_jira_data_btn"):
                    if st.session_state.selected_jira_project_keys:
                        with st.spinner("Fetching data from selected Jira projects..."):
                            jira_context = format_jira_data(st.session_state.jira_client, st.session_state.selected_jira_project_keys)
                        st.session_state.project_context = jira_context.strip() if jira_context else ""
                        if st.session_state.project_context:
                            st.success("Jira project data updated!")
                        else:
                            st.warning("No data fetched or an error occurred. Check project permissions or selected issues.")
                    else:
                        st.session_state.project_context = ""
                        st.warning("No Jira projects selected. Context cleared.")

            # --- CORRECTED PREVIEW SECTION ---
            # Display preview if context exists (NO inner expander needed)
            if st.session_state.get("project_context"):
                st.markdown("---") # Optional separator
                st.caption("Preview Fetched Jira Context:") # Add a caption for clarity
                st.text_area(
                    "jira_ctx_preview_sidebar", # Use key as first arg
                    value=st.session_state.project_context,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed" # Hide default label
                )
            # --- END OF CORRECTION ---

    st.divider()

    # --- RAG Document Sources ---
    st.header("RAG Document Context")
    st.session_state.use_rag_context = st.toggle(
        "Use Document Context (RAG)",
        value=st.session_state.get("use_rag_context", True),
        key="toggle_rag_context"
        )

    # Display Indexed Files Info
    rag_files_info = st.session_state.get("uploaded_files_info", [])
    if st.session_state.get("rag_index_ready", False) and rag_files_info:
        st.markdown("**Indexed Documents:**")
        for i, file_info in enumerate(rag_files_info):
             name = file_info.get('name', 'Unknown')
             source = file_info.get('source', 'Unknown source')
             st.markdown(f"<span class='small-font'>- {name} ({source})</span>", unsafe_allow_html=True)
        if st.button("Clear RAG Index", key="clear_rag_button_sidebar"):
             st.session_state.vector_store = None
             st.session_state.rag_index_ready = False
             st.session_state.uploaded_files_info = []
             st.success("RAG index cleared.")
             st.rerun()
    elif st.session_state.get("processing_docs", False):
         # Spinner shown in main processing block
         st.info("Processing selected documents...")
    else:
        st.info("No documents currently indexed for RAG.")

    # --- File Uploader ---
    with st.expander("Upload Local Files", expanded=True):
        # Assign uploaded files to a variable accessible outside the expander
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, MD, CSV, XLSX files",
            type=SUPPORTED_DOC_TYPES,
            accept_multiple_files=True,
            key="rag_file_uploader_sidebar",
            help="Upload documents to be indexed for RAG."
        )

    # --- Cloud Drive Connectors ---
    with st.expander("Connect Cloud Drives", expanded=False):

        # --- Google Drive (Modified Web Flow) ---
        st.subheader("Google Drive")
        gdrive_service, gdrive_error_msg = get_google_drive_service()

        if gdrive_service:
            st.success("Google Drive Connected")
            if st.button("Disconnect Google Drive", key="gdrive_disconnect_btn"):
                st.session_state.google_creds = None
                st.session_state.google_service = None
                st.session_state.google_files = None
                st.success("Disconnected from Google Drive.")
                if "code" in st.query_params: st.query_params.clear()
                st.rerun()

            if st.button("List Google Drive Files", key="gdrive_list_files_btn"):
                 with st.spinner("Fetching Google Drive files..."):
                    files, error = list_google_drive_files(gdrive_service)
                    if files is not None:
                        st.session_state.google_files = files
                        if not files: st.info("No files found in your Google Drive root (folders excluded).")
                    else:
                        st.error(f"Failed to list Google Drive files: {error}")

        elif gdrive_error_msg:
             if "Authentication required" not in gdrive_error_msg:
                 st.error(f"Google Drive connection issue: {gdrive_error_msg}")
             # Auth link shown by get_google_drive_service
        else:
             st.info("Checking Google Drive connection status...")

        st.divider()

        # --- OneDrive (Existing Device Flow) ---
        st.subheader("OneDrive")
        ms_account = st.session_state.get("ms_account")
        if ms_account:
            st.success(f"OneDrive Connected as: {ms_account.get('username', 'Unknown')}")
            if st.button("Disconnect OneDrive", key="msdrive_disconnect_btn"):
                 st.session_state.ms_token = None
                 st.session_state.ms_account = None
                 st.session_state.ms_files = None
                 if "_ms_waiting_for_signin_check" in st.session_state: del st.session_state._ms_waiting_for_signin_check
                 st.success("Disconnected from OneDrive.")
                 st.rerun()

            if st.button("List OneDrive Files", key="msdrive_list_files_btn"):
                access_token_info = st.session_state.get("ms_token")
                access_token = access_token_info.get("access_token") if isinstance(access_token_info, dict) else None
                if access_token:
                     with st.spinner("Fetching OneDrive files..."):
                        files, error = list_onedrive_files(access_token)
                        if files is not None:
                            st.session_state.ms_files = files
                            if not files: st.info("No files found in your OneDrive root (folders excluded).")
                        else:
                            st.error(f"Failed to list OneDrive files: {error}")
                else:
                     st.warning("OneDrive token missing or expired. Please connect/reconnect.")
        else:
            st.info("Connect to OneDrive to browse files.")
            if st.button("Connect OneDrive", key="msdrive_connect_btn"):
                pca = get_ms_graph_client()
                if pca:
                    with st.spinner("Initiating Microsoft sign-in..."):
                        token, account, error = acquire_ms_token_interactive(pca)
                        if token and account: st.rerun()
                        elif error and "Waiting" not in error: st.rerun()
                        # No rerun if waiting

    # --- File Selection from Drives ---
    # Calculate selected files here so button logic below can use it
    selected_drive_files_info = []

    # Google Drive File Selection
    gdrive_files_list = st.session_state.get("google_files")
    if isinstance(gdrive_files_list, list):
        with st.expander("Select Google Drive Files to Add", expanded=bool(gdrive_files_list)):
            gdrive_options = {f"g_{f['id']}": f['name'] for f in gdrive_files_list if 'id' in f and 'name' in f}
            if gdrive_options:
                selected_gdrive_keys = st.multiselect(
                    "Google Drive Files:", options=list(gdrive_options.keys()),
                    format_func=lambda file_key: gdrive_options.get(file_key, "Invalid Key"),
                    key="gdrive_multiselect_sidebar"
                )
                selected_drive_files_info.extend([
                    {"id": fkey.split("_")[1], "name": gdrive_options[fkey], "source": "google"}
                    for fkey in selected_gdrive_keys if fkey in gdrive_options
                ])
            else: st.caption("No Google Drive files listed. Click 'List Google Drive Files' above.")

    # OneDrive File Selection
    msdrive_files_list = st.session_state.get("ms_files")
    if isinstance(msdrive_files_list, list):
         with st.expander("Select OneDrive Files to Add", expanded=bool(msdrive_files_list)):
            msdrive_options = {f"ms_{f['id']}": f['name'] for f in msdrive_files_list if 'id' in f and 'name' in f}
            if msdrive_options:
                selected_msdrive_keys = st.multiselect(
                    "OneDrive Files:", options=list(msdrive_options.keys()),
                    format_func=lambda file_key: msdrive_options.get(file_key, "Invalid Key"),
                    key="msdrive_multiselect_sidebar"
                )
                selected_drive_files_info.extend([
                    {"id": fkey.split("_")[1], "name": msdrive_options[fkey], "source": "onedrive"}
                    for fkey in selected_msdrive_keys if fkey in msdrive_options
                ])
            else: st.caption("No OneDrive files listed. Click 'List OneDrive Files' above.")


    # --- Button to Add Selected Files to Index (CORRECTED LOGIC) ---
    # Determine if there are files ready to be processed
    # Use the 'uploaded_files' variable defined within the sidebar scope
    has_local_files = bool(uploaded_files)
    has_drive_files = bool(selected_drive_files_info)

    # Enable button only if files are selected AND not already processing
    button_disabled = st.session_state.get("processing_docs", False) or not (has_local_files or has_drive_files)
    if st.button("Process & Add Selected Files to RAG Index", key="process_files_button_sidebar", disabled=button_disabled):
        if not (has_local_files or has_drive_files):
             st.warning("Please select at least one file to process.") # Failsafe
        else:
            st.session_state.processing_docs = True
            # Store drive file selections temporarily BEFORE rerunning
            st.session_state._temp_drive_files_to_process = selected_drive_files_info
            # Store references to local uploaded files (if possible, state management is tricky)
            # Storing the whole list might work for one rerun cycle
            st.session_state._temp_local_files_to_process = uploaded_files

            logging.info(f"Button clicked. processing_docs=True. Storing {len(selected_drive_files_info)} drive files and {len(uploaded_files)} local files for processing.")
            st.rerun() # Rerun immediately to show spinner and execute processing block

# --- Separate Block for Processing Logic (runs on rerun if processing_docs is True) ---
if st.session_state.get("processing_docs", False):
    # Retrieve the stored lists of files
    drive_files_to_process = st.session_state.get("_temp_drive_files_to_process", [])
    # Retrieve local files stored temporarily
    files_to_process_local = st.session_state.get("_temp_local_files_to_process", [])

    logging.info(f"Processing block entered. Will process {len(files_to_process_local)} local files and {len(drive_files_to_process)} drive files.")

    # Check if there's anything to process
    if not files_to_process_local and not drive_files_to_process:
         st.warning("No files found to process in this cycle. Please select files and try again.")
         st.session_state.processing_docs = False # Reset flag
         # Clean up temp state
         if "_temp_drive_files_to_process" in st.session_state: del st.session_state._temp_drive_files_to_process
         if "_temp_local_files_to_process" in st.session_state: del st.session_state._temp_local_files_to_process
         st.rerun()
    else:
        # Proceed with the processing logic
        with st.spinner("Processing files... This may take some time."):
            all_docs = []
            processed_files_info = [] # Track successfully processed files
            has_errors = False

            # 1. Process local uploads
            if files_to_process_local:
                st.info(f"Processing {len(files_to_process_local)} local file(s)...")
                for up_file in files_to_process_local:
                    if hasattr(up_file, 'name') and hasattr(up_file, 'getvalue'):
                        try:
                            logging.info(f"Loading local file: {up_file.name}")
                            # up_file.seek(0) # Not always necessary with getvalue()
                            docs = load_document(up_file)
                            if docs:
                                all_docs.extend(docs)
                                processed_files_info.append({"name": up_file.name, "source": "local"})
                            else:
                                has_errors = True # load_document handles st.error
                                logging.warning(f"load_document returned None for local file: {up_file.name}")
                        except Exception as local_e:
                             has_errors = True
                             st.error(f"Failed to process local file {up_file.name}: {local_e}")
                             logging.error(f"Error processing local file {up_file.name}", exc_info=True)
                    else:
                         logging.warning(f"Skipping unexpected item in local file list: {type(up_file)}")

            # 2. Process selected drive files (using stored info)
            if drive_files_to_process:
                 st.info(f"Processing {len(drive_files_to_process)} file(s) from cloud drives...")
                 for drive_file in drive_files_to_process:
                    file_id = drive_file['id']
                    file_name = drive_file['name']
                    source = drive_file['source']
                    logging.info(f"Downloading {source} file: {file_name} ({file_id})")
                    file_content_obj = None; error_msg = None

                    try:
                        # Check connection status before downloading
                        if source == "google":
                            g_service = st.session_state.get("google_service")
                            if g_service: file_content_obj, error_msg = download_google_drive_file(g_service, file_id, file_name)
                            else: error_msg = "Google Drive disconnected."
                        elif source == "onedrive":
                            ms_token_info = st.session_state.get("ms_token")
                            access_token = ms_token_info.get("access_token") if isinstance(ms_token_info, dict) else None
                            if access_token: file_content_obj, error_msg = download_onedrive_file(access_token, file_id, file_name)
                            else: error_msg = "OneDrive disconnected or token invalid."

                        if file_content_obj:
                            logging.info(f"Loading downloaded file content: {file_name}")
                            docs = load_document(file_content_obj)
                            if docs:
                                all_docs.extend(docs)
                                processed_files_info.append({"name": file_name, "source": source})
                            else:
                                has_errors = True # load_document handles st.error
                                logging.warning(f"load_document returned None for downloaded file: {file_name}")
                            if hasattr(file_content_obj, 'close'): file_content_obj.close() # Close BytesIO
                        else:
                            has_errors = True
                            st.error(f"Failed to download {file_name} from {source}: {error_msg}")
                            logging.error(f"Download failed for {source} file {file_name} ({file_id}): {error_msg}")

                    except Exception as drive_e:
                         has_errors = True
                         st.error(f"Failed to process drive file {file_name}: {drive_e}")
                         logging.error(f"Error processing drive file {file_name}", exc_info=True)


            # 3. Split documents
            chunks = []
            if all_docs:
                with st.spinner("Splitting documents into chunks..."):
                    chunks = split_documents(all_docs)

            # 4. Create or Update Vector Store
            if chunks:
                new_vector_store = create_vector_store(chunks)
                if new_vector_store:
                    existing_store = st.session_state.get("vector_store")
                    if existing_store is not None:
                        st.warning("Replacing existing RAG index with newly processed files.")
                        # Add merging logic here if desired and FAISS version supports it
                    st.session_state.vector_store = new_vector_store
                    st.session_state.uploaded_files_info = processed_files_info # Update with successfully processed
                    st.session_state.rag_index_ready = True
                    st.success(f"Successfully processed and indexed {len(processed_files_info)} file(s). RAG is ready.")
                    if has_errors:
                        st.warning("Some files could not be processed. Check logs or errors above.")
                else:
                    st.error("Failed to create vector store from processed documents. Index not updated.")
            elif has_errors:
                 st.error("Document processing completed with errors. No new index created or updated.")
            elif not all_docs and (files_to_process_local or drive_files_to_process):
                 st.warning("No valid content could be extracted from the selected files. Index not updated.")
            # No message needed if no files were selected initially

            # Reset processing flag and clear temporary state
            st.session_state.processing_docs = False
            if "_temp_drive_files_to_process" in st.session_state: del st.session_state._temp_drive_files_to_process
            if "_temp_local_files_to_process" in st.session_state: del st.session_state._temp_local_files_to_process
            # Clear the file uploader state by assigning an empty list to its key
            # st.session_state.rag_file_uploader_sidebar = []
            st.rerun() # Final rerun to update UI after processing is fully done


# --- Sidebar Part 2: Personality Config (runs after potential processing block rerun) ---
with st.sidebar:
    # Ensure divider doesn't appear mid-processing
    if not st.session_state.get("processing_docs", False):
        st.divider()
        # --- Personality Configuration ---
        st.header("Personality Configuration")
        personality_type = st.radio("Select Personality Framework:", ["MBTI", "Big Five", "Both"], index=2, key="personality_framework_radio")
        st.session_state.use_big_five = personality_type in ["Big Five", "Both"]
        use_mbti = personality_type in ["MBTI", "Both"]

        # --- Persona Selection & Manual Config ---
        st.subheader("Left Model Profile")
        left_persona_options = ["Manual Customization"] + list(PERSONAS.keys())
        left_persona_choice = st.session_state.get("left_persona_choice", "Manual Customization")
        left_persona_idx = left_persona_options.index(left_persona_choice) if left_persona_choice in left_persona_options else 0
        left_persona = st.selectbox("Select Persona (Left):", left_persona_options, index=left_persona_idx, key="left_persona_choice")

        if left_persona != "Manual Customization":
            persona_data = PERSONAS[left_persona]
            st.info(f"Bio: {persona_data['Quick Bio']} | MBTI: {persona_data['MBTI']}")
            st.session_state.left_mbti = persona_data["MBTI"]
            st.session_state.left_big_five = persona_data["Big Five"].copy()
            with st.expander("View Big Five Traits (Left)", expanded=False):
                for trait, value in st.session_state.left_big_five.items():
                    st.slider(f"{trait.capitalize()} (Left)", 0, 100, value, key=f"left_{trait}_view_disp", disabled=True)
        else: # Manual customization enabled
            if use_mbti:
                mbti_default_l = st.session_state.get("left_mbti", MBTI_TYPES[0])
                mbti_idx_l = MBTI_TYPES.index(mbti_default_l) if mbti_default_l in MBTI_TYPES else 0
                st.session_state.left_mbti = st.selectbox("MBTI (Left):", MBTI_TYPES, index=mbti_idx_l, key="left_mbti_select_manual")
            if st.session_state.use_big_five:
                bf_defaults_l = st.session_state.get("left_big_five", {"openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50, "neuroticism": 50})
                with st.expander("Adjust Big Five Traits (Left)", expanded=True):
                    st.session_state.left_big_five["openness"] = st.slider("Openness (Left)", 0, 100, bf_defaults_l.get("openness", 50), key="left_openness_manual")
                    st.session_state.left_big_five["conscientiousness"] = st.slider("Conscientiousness (Left)", 0, 100, bf_defaults_l.get("conscientiousness", 50), key="left_conscientiousness_manual")
                    st.session_state.left_big_five["extraversion"] = st.slider("Extraversion (Left)", 0, 100, bf_defaults_l.get("extraversion", 50), key="left_extraversion_manual")
                    st.session_state.left_big_five["agreeableness"] = st.slider("Agreeableness (Left)", 0, 100, bf_defaults_l.get("agreeableness", 50), key="left_agreeableness_manual")
                    st.session_state.left_big_five["neuroticism"] = st.slider("Neuroticism (Left)", 0, 100, bf_defaults_l.get("neuroticism", 50), key="left_neuroticism_manual")

        st.subheader("Right Model Profile")
        right_persona_options = ["Manual Customization"] + list(PERSONAS.keys())
        right_persona_choice = st.session_state.get("right_persona_choice", "Manual Customization")
        right_persona_idx = right_persona_options.index(right_persona_choice) if right_persona_choice in right_persona_options else 0
        right_persona = st.selectbox("Select Persona (Right):", right_persona_options, index=right_persona_idx, key="right_persona_choice")

        if right_persona != "Manual Customization":
            persona_data = PERSONAS[right_persona]
            st.info(f"Bio: {persona_data['Quick Bio']} | MBTI: {persona_data['MBTI']}")
            st.session_state.right_mbti = persona_data["MBTI"]
            st.session_state.right_big_five = persona_data["Big Five"].copy()
            with st.expander("View Big Five Traits (Right)", expanded=False):
                for trait, value in st.session_state.right_big_five.items():
                    st.slider(f"{trait.capitalize()} (Right)", 0, 100, value, key=f"right_{trait}_view_disp", disabled=True)
        else: # Manual customization enabled
            if use_mbti:
                mbti_default_r = st.session_state.get("right_mbti", MBTI_TYPES[0])
                mbti_idx_r = MBTI_TYPES.index(mbti_default_r) if mbti_default_r in MBTI_TYPES else 0
                st.session_state.right_mbti = st.selectbox("MBTI (Right):", MBTI_TYPES, index=mbti_idx_r, key="right_mbti_select_manual")
            if st.session_state.use_big_five:
                bf_defaults_r = st.session_state.get("right_big_five", {"openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50, "neuroticism": 50})
                with st.expander("Adjust Big Five Traits (Right)", expanded=True):
                    st.session_state.right_big_five["openness"] = st.slider("Openness (Right)", 0, 100, bf_defaults_r.get("openness", 50), key="right_openness_manual")
                    st.session_state.right_big_five["conscientiousness"] = st.slider("Conscientiousness (Right)", 0, 100, bf_defaults_r.get("conscientiousness", 50), key="right_conscientiousness_manual")
                    st.session_state.right_big_five["extraversion"] = st.slider("Extraversion (Right)", 0, 100, bf_defaults_r.get("extraversion", 50), key="right_extraversion_manual")
                    st.session_state.right_big_five["agreeableness"] = st.slider("Agreeableness (Right)", 0, 100, bf_defaults_r.get("agreeableness", 50), key="right_agreeableness_manual")
                    st.session_state.right_big_five["neuroticism"] = st.slider("Neuroticism (Right)", 0, 100, bf_defaults_r.get("neuroticism", 50), key="right_neuroticism_manual")

        st.divider()
        st.subheader("About")
        st.write("Compare models with custom personalities and context from Jira & RAG.")


# --- Main Chat Interface ---

# --- Function to get model response (Includes RAG) ---
# Definition of get_model_response remains the same as the previous full version
def get_model_response(messages, model, mbti, big_five, use_mbti, use_big_five,
                       jira_context=None, use_jira_context=False,
                       rag_retriever=None, use_rag=False, user_query=""):
    """ Obtains model response, incorporating context and personality. """
    try:
        # 0. Pre-checks
        openai_api_key = st.session_state.get("OPENAI_API_KEY")
        if not openai_api_key:
            return "Error: Please enter your OpenAI API key in the sidebar.", None
        if not openai.api_key: # Ensure openai client knows the key
             openai.api_key = openai_api_key

        if not user_query and messages and isinstance(messages, list):
             last_user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), None)
             if last_user_msg: user_query = last_user_msg

        # 1. Determine Model Name
        official_model_name = AVAILABLE_FINE_TUNED_MODELS.get(model, model)

        # 2. Initialize System Prompt Parts
        system_prompt_parts = []
        full_context_for_display = [] # For showing context used in UI (optional)

        # 3. Add RAG Context (if enabled and retriever available)
        rag_context_str = ""
        if use_rag and rag_retriever and user_query:
            logging.info(f"Attempting RAG retrieval for query: '{user_query[:50]}...'")
            with st.spinner(f"Retrieving relevant documents for {model}..."): # Spinner within function call
                try:
                    retrieved_docs = rag_retriever.invoke(user_query) # Use invoke for LCEL Retrievers
                    if retrieved_docs:
                        rag_context_str = format_retrieved_docs(retrieved_docs)
                        if rag_context_str and "No relevant documents found" not in rag_context_str :
                            system_prompt_parts.append("=== Context from Your Documents (RAG) ===")
                            system_prompt_parts.append(rag_context_str)
                            system_prompt_parts.append("==========================================")
                            full_context_for_display.append(rag_context_str)
                            logging.info(f"Added RAG context ({len(retrieved_docs)} chunks) for model {model}")
                    else:
                         logging.info("RAG retrieval returned no documents.")

                except Exception as rag_e:
                    logging.error(f"Error during RAG retrieval for {model}: {rag_e}", exc_info=True)
                    st.warning(f"Could not retrieve RAG context for {model}: {rag_e}") # Show warning in UI

        # 4. Add Jira Project Context (if enabled and available)
        if use_jira_context and jira_context:
            system_prompt_parts.append("=== Context from Jira ===")
            system_prompt_parts.append(jira_context)
            system_prompt_parts.append("=========================")
            full_context_for_display.append(jira_context)
            logging.info(f"Added Jira context for model {model}")

        # 5. Add Personality Profile
        personality_description = create_personality_profile(mbti, big_five, use_mbti, use_big_five)
        if personality_description:
            system_prompt_parts.append("=== Your Personality Profile (Adapt response style) ===")
            system_prompt_parts.append(personality_description)
            system_prompt_parts.append("=====================================================")
            full_context_for_display.append(f"Personality Profile:\n{personality_description}")


        # 6. Add Base Instructions / Fine-Tuning Prompt
        if model in AVAILABLE_FINE_TUNED_MODELS:
            special_ft_prompt = FINE_TUNED_PROMPTS.get(model, DEFAULT_SPECIAL_FT_PROMPT)
            system_prompt_parts.append("=== Model Instructions ===")
            system_prompt_parts.append(special_ft_prompt)
            system_prompt_parts.append("==========================")
        elif model in AVAILABLE_MODELS:
            system_prompt_parts.append("You are a helpful project management assistant. Respond clearly and concisely.")

        # Combine system prompt parts
        system_prompt = "\n\n".join(filter(None, system_prompt_parts)).strip()

        # 7. Prepare messages for API call
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (user and assistant roles only)
        if messages and isinstance(messages, list):
            history_for_api = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages if msg["role"] in ["user", "assistant"]
            ]
            # Ensure the very last message is the user query if not already included
            if not history_for_api or history_for_api[-1].get("role") != "user":
                 if messages and messages[-1]["role"] == "user":
                     history_for_api.append({"role": "user", "content": messages[-1]["content"]})

            api_messages.extend(history_for_api)

        # Ensure there's at least one user message if history was provided
        if not any(msg['role'] == 'user' for msg in api_messages):
             logging.error("No user message found in history to send to API.")
             return "Error: Cannot generate response without user input in the conversation.", system_prompt

        # 8. Make API call
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            logging.info(f"Calling OpenAI API. Model: {official_model_name}. System prompt length: {len(system_prompt)}. History messages: {len(api_messages) - (1 if system_prompt else 0)}")

            response = client.chat.completions.create(
                model=official_model_name,
                messages=api_messages,
                temperature=0.7,
                # max_tokens=1500,
            )
            response_content = response.choices[0].message.content
            logging.info(f"Received response from {model}. Length: {len(response_content)}")
            return response_content, system_prompt

        except openai.AuthenticationError:
            logging.error("OpenAI Authentication Error - Check API Key")
            st.error("OpenAI Authentication failed. Please check your API key in the sidebar.")
            return "Error: OpenAI Authentication failed.", system_prompt
        except openai.RateLimitError:
            logging.warning("OpenAI Rate Limit Exceeded")
            st.warning("OpenAI Rate limit exceeded. Please try again later or check your usage.")
            return "Error: OpenAI Rate limit exceeded.", system_prompt
        except openai.APIConnectionError as e:
            logging.error(f"OpenAI API Connection Error: {e}")
            st.error(f"Could not connect to OpenAI API: {e}")
            return f"Error: Could not connect to OpenAI. {e}", system_prompt
        except openai.BadRequestError as e:
             logging.error(f"OpenAI Bad Request Error (likely context length): {e}")
             st.error(f"OpenAI Request Error: {e}. The context or conversation history might be too long.")
             return f"Error: OpenAI Request Failed ({e}). Context may be too long.", system_prompt
        except Exception as e:
            logging.error(f"Error getting model response for {model}: {e}", exc_info=True)
            st.error(f"An unexpected error occurred while generating the response for {model}: {e}")
            return f"Error generating response: {str(e)}", system_prompt

    except Exception as outer_e:
         logging.error(f"Error preparing model response for {model}: {outer_e}", exc_info=True)
         st.error(f"Failed to prepare request for {model}: {outer_e}")
         return f"Error setting up request: {str(outer_e)}", None


# --- Define Chat Columns ---
col1, col2 = st.columns(2)

# Retrieve RAG retriever if index is ready
current_rag_retriever = None
if st.session_state.get("rag_index_ready", False) and st.session_state.get("vector_store"):
    current_rag_retriever = get_retriever(st.session_state.vector_store)
    if not current_rag_retriever:
         st.warning("RAG index is marked ready, but failed to create retriever.", icon="⚠️")


# --- Left Column Setup ---
with col1:
    left_model_options = ALL_MODELS
    left_model_default = st.session_state.get("left_model", left_model_options[0])
    left_model_idx = left_model_options.index(left_model_default) if left_model_default in left_model_options else 0
    left_model = st.selectbox(
        "Select Left Model:", left_model_options, index=left_model_idx, key="left_model_select_main"
    )
    st.session_state.left_model = left_model

    use_mbti_l = st.session_state.get("personality_framework_radio", "Both") in ["MBTI", "Both"]
    use_big_five_l = st.session_state.get("personality_framework_radio", "Both") in ["Big Five", "Both"]
    personality_framework_l = "MBTI" if use_mbti_l and not use_big_five_l else "Big Five" if use_big_five_l and not use_mbti_l else "MBTI & Big Five"
    profile_html_l = create_personality_profile(st.session_state.get("left_mbti"), st.session_state.get("left_big_five"), use_mbti_l, use_big_five_l).replace("\n", "<br>")
    st.markdown(f"<div class='personality-section'><strong>Personality ({personality_framework_l}):</strong><br>{profile_html_l if profile_html_l else 'Default'}</div>", unsafe_allow_html=True)

    st.subheader(f"Chat with: {left_model}")
    left_chat_container = st.container()
    with left_chat_container:
        st.markdown("<div class='chat-container' id='left-chat'>", unsafe_allow_html=True)
        for message in st.session_state.get("left_messages", []):
            if message["role"] == "assistant": message["model_name"] = left_model
            display_chat_message(message)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Right Column Setup ---
with col2:
    right_model_options = ALL_MODELS
    right_model_default = st.session_state.get("right_model", right_model_options[1] if len(right_model_options)>1 else right_model_options[0])
    right_model_idx = right_model_options.index(right_model_default) if right_model_default in right_model_options else 0
    right_model = st.selectbox(
        "Select Right Model:", right_model_options, index=right_model_idx, key="right_model_select_main"
    )
    st.session_state.right_model = right_model

    use_mbti_r = st.session_state.get("personality_framework_radio", "Both") in ["MBTI", "Both"]
    use_big_five_r = st.session_state.get("personality_framework_radio", "Both") in ["Big Five", "Both"]
    personality_framework_r = "MBTI" if use_mbti_r and not use_big_five_r else "Big Five" if use_big_five_r and not use_mbti_r else "MBTI & Big Five"
    profile_html_r = create_personality_profile(st.session_state.get("right_mbti"), st.session_state.get("right_big_five"), use_mbti_r, use_big_five_r).replace("\n", "<br>")
    st.markdown(f"<div class='personality-section'><strong>Personality ({personality_framework_r}):</strong><br>{profile_html_r if profile_html_r else 'Default'}</div>", unsafe_allow_html=True)

    st.subheader(f"Chat with: {right_model}")
    right_chat_container = st.container()
    with right_chat_container:
        st.markdown("<div class='chat-container' id='right-chat'>", unsafe_allow_html=True)
        for message in st.session_state.get("right_messages", []):
             if message["role"] == "assistant": message["model_name"] = right_model
             display_chat_message(message)
        st.markdown("</div>", unsafe_allow_html=True)

# --- User Input Area ---
user_input = st.text_area("Your message:", height=100, key="user_message_input_main")

# --- Submit Button Row ---
col_submit, col_clear_left, col_clear_right, col_clear_both = st.columns([3, 1, 1, 1])

with col_submit:
    if st.button("Submit to Both Models", key="submit_button_main"):
        openai_key = st.session_state.get("OPENAI_API_KEY")
        if not openai_key:
             st.warning("Please enter your OpenAI API key in the sidebar first.")
        elif not user_input.strip():
             st.warning("Please enter a message.")
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            user_message_formatted = format_message("user", user_input, timestamp=timestamp)

            # Add user message immediately to state for display
            st.session_state.setdefault("left_messages", []).append(user_message_formatted)
            st.session_state.setdefault("right_messages", []).append(user_message_formatted)

            # Set flag and store query for processing on rerun
            st.session_state._processing_response = True
            st.session_state._user_query_for_processing = user_input
            st.rerun() # Trigger rerun to show user message and spinners

# --- Processing Logic on Rerun ---
if st.session_state.get("_processing_response"):
    user_query = st.session_state.get("_user_query_for_processing", "")
    timestamp = datetime.now().strftime("%H:%M:%S") # Get new timestamp for response

    # Retrieve contexts and settings needed for generation
    use_jira = st.session_state.get("use_project_context", False)
    jira_ctx = st.session_state.get("project_context") if use_jira else None
    use_rag = st.session_state.get("use_rag_context", False)
    rag_retriever = current_rag_retriever # Use retriever obtained at the start of the script run

    use_mbti_global = st.session_state.get("personality_framework_radio", "Both") in ["MBTI", "Both"]
    use_big_five_global = st.session_state.get("personality_framework_radio", "Both") in ["Big Five", "Both"]

    left_model_to_use = st.session_state.get("left_model")
    right_model_to_use = st.session_state.get("right_model")

    # --- Left Model Call ---
    left_response_content, left_sys_prompt = "Error: Processing failed.", None
    # Show spinner using columns context if possible, otherwise just log
    spinner_placeholder_left = col1.empty() # Use placeholder in main layout
    with spinner_placeholder_left:
        with st.spinner(f"Generating response from {left_model_to_use}..."):
            try:
                left_response_content, left_sys_prompt = get_model_response(
                    messages=st.session_state.get("left_messages", []),
                    model=left_model_to_use,
                    mbti=st.session_state.get("left_mbti"),
                    big_five=st.session_state.get("left_big_five"),
                    use_mbti=use_mbti_global, use_big_five=use_big_five_global,
                    jira_context=jira_ctx, use_jira_context=use_jira,
                    rag_retriever=rag_retriever, use_rag=use_rag,
                    user_query=user_query
                )
            except Exception as e:
                left_response_content = f"Error during left model generation: {e}"
                logging.error(f"Unhandled error calling get_model_response (Left): {e}", exc_info=True)
    spinner_placeholder_left.empty() # Clear spinner placeholder

    # Add left assistant response
    left_assistant_msg = format_message(
        role="assistant", content=left_response_content,
        model_name=left_model_to_use, timestamp=timestamp)
    st.session_state.setdefault("left_messages", []).append(left_assistant_msg)

    # --- Right Model Call ---
    right_response_content, right_sys_prompt = "Error: Processing failed.", None
    spinner_placeholder_right = col2.empty()
    with spinner_placeholder_right:
        with st.spinner(f"Generating response from {right_model_to_use}..."):
            try:
                right_response_content, right_sys_prompt = get_model_response(
                    messages=st.session_state.get("right_messages", []),
                    model=right_model_to_use,
                    mbti=st.session_state.get("right_mbti"),
                    big_five=st.session_state.get("right_big_five"),
                    use_mbti=use_mbti_global, use_big_five=use_big_five_global,
                    jira_context=jira_ctx, use_jira_context=use_jira,
                    rag_retriever=rag_retriever, use_rag=use_rag,
                    user_query=user_query
                )
            except Exception as e:
                 right_response_content = f"Error during right model generation: {e}"
                 logging.error(f"Unhandled error calling get_model_response (Right): {e}", exc_info=True)
    spinner_placeholder_right.empty()

    # Add right assistant response
    right_assistant_msg = format_message(
        role="assistant", content=right_response_content,
        model_name=right_model_to_use, timestamp=timestamp)
    st.session_state.setdefault("right_messages", []).append(right_assistant_msg)

    # Clear processing flag and stored query
    del st.session_state._processing_response
    if "_user_query_for_processing" in st.session_state:
         del st.session_state._user_query_for_processing
    st.rerun() # Final rerun to display results


# --- Clear Buttons ---
with col_clear_left:
    if st.button("Clear Left", key="clear_left_btn_main"):
        st.session_state.left_messages = []
        st.rerun()

with col_clear_right:
    if st.button("Clear Right", key="clear_right_btn_main"):
        st.session_state.right_messages = []
        st.rerun()

with col_clear_both:
    if st.button("Clear Both", key="clear_both_btn_main"):
        st.session_state.left_messages = []
        st.session_state.right_messages = []
        st.rerun()

# --- Footer or Debug Info (Optional) ---
# with st.expander("Debug Session State"):
#     st.write(st.session_state)