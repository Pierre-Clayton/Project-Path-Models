# utils/drive_connectors.py
import streamlit as st
import os
import pickle
import logging
import requests
import msal
from io import BytesIO

# Google Drive Imports
from google_auth_oauthlib.flow import Flow  # Use the base Flow for web
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import googleapiclient.http # For download

# Local Imports
from .config import GOOGLE_SCOPES, MS_CLIENT_ID, MS_AUTHORITY, MS_SCOPES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Drive Functions ---

def get_google_client_config():
    """Loads Google OAuth config from Streamlit secrets."""
    try:
        client_id = st.secrets["GOOGLE_CLIENT_ID"]
        client_secret = st.secrets["GOOGLE_CLIENT_SECRET"]
        redirect_uri = st.secrets["GOOGLE_REDIRECT_URI"]

        if not client_id or not client_secret or not redirect_uri:
            logging.error("Google OAuth secrets (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI) are missing in secrets.toml.")
            st.error("Google OAuth secrets (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI) are missing in secrets.toml.")
            return None, None # Return two Nones

        # Format expected by google_auth_oauthlib.flow.Flow for 'web'
        client_config = {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [redirect_uri], # Must contain the one used below
                 # Optional but recommended for web flows
                # Extract base URL like 'http://localhost:8501' -> 'http://localhost:8501'
                # Or 'https://myapp.streamlit.app/' -> 'https://myapp.streamlit.app'
                "javascript_origins": [redirect_uri.rsplit('/', 1)[0] if '/' in redirect_uri else redirect_uri]
            }
        }
        logging.info(f"Loaded Google Client Config for redirect URI: {redirect_uri}")
        return client_config, redirect_uri
    except KeyError as e:
        logging.error(f"Missing Google OAuth secret: {e}. Please configure GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in .streamlit/secrets.toml")
        st.error(f"Missing Google OAuth secret: {e}. Please configure GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in .streamlit/secrets.toml")
        return None, None
    except Exception as e:
        logging.error(f"Error loading Google config from secrets: {e}", exc_info=True)
        st.error(f"Error loading Google config from secrets: {e}")
        return None, None


def get_google_drive_service():
    """
    Handles Google Drive authentication using web flow and Streamlit secrets.
    Manages the redirect using st.query_params.
    Returns the service object or None, and an error message string or None.
    """
    creds = st.session_state.get("google_creds")
    service = st.session_state.get("google_service")

    # If we already have a valid service, return it
    if service and creds and isinstance(creds, Credentials) and creds.valid:
        # logging.debug("Returning existing valid Google Drive service.")
        return service, None

    # Check if credentials exist and are valid, try refreshing if expired
    if creds and isinstance(creds, Credentials) and not creds.valid:
        logging.info("Google credentials found but invalid/expired.")
        if creds.expired and creds.refresh_token:
            try:
                logging.info("Attempting to refresh Google credentials...")
                creds.refresh(Request())
                st.session_state.google_creds = creds # Update session state
                logging.info("Google credentials refreshed successfully.")
                # Need to rebuild service after refresh
                try:
                    service = build('drive', 'v3', credentials=creds)
                    st.session_state.google_service = service
                    logging.info("Google Drive service rebuilt successfully after refresh.")
                    return service, None
                except Exception as build_err:
                    logging.error(f"Failed to build Google Drive service after refresh: {build_err}", exc_info=True)
                    st.error(f"Failed to rebuild Google Drive service after refresh: {build_err}")
                    st.session_state.google_service = None # Clear invalid service attempt
                    return None, "Failed to rebuild service after refresh."

            except Exception as e:
                logging.warning(f"Failed to refresh Google token: {e}. Re-authentication needed.")
                st.warning(f"Could not refresh Google credentials ({e}). Please re-authenticate.")
                # Clear creds to force re-auth
                st.session_state.google_creds = None
                st.session_state.google_service = None
                creds = None
                service = None
        else:
            # Invalid creds without refresh token mean re-auth needed
            logging.info("Invalid Google credentials without refresh token. Clearing state.")
            st.session_state.google_creds = None
            st.session_state.google_service = None
            creds = None
            service = None

    # --- Authentication Flow ---
    client_config_data = get_google_client_config()
    if not client_config_data or not client_config_data[0]:
        # Error displayed in get_google_client_config
        return None, "Google OAuth secrets not configured."

    client_config, redirect_uri = client_config_data

    # Ensure Flow uses the correct structure for client_config
    try:
        flow = Flow.from_client_config( # Use from_client_config for dictionary
            client_config=client_config,
            scopes=GOOGLE_SCOPES,
            redirect_uri=redirect_uri
        )
    except Exception as flow_init_error:
         logging.error(f"Failed to initialize Google OAuth Flow: {flow_init_error}", exc_info=True)
         st.error(f"Failed to initialize Google OAuth Flow: {flow_init_error}")
         return None, "OAuth Flow initialization error."

    # Check if the user was just redirected back with an authorization code
    query_params = st.query_params.to_dict()
    auth_code = query_params.get("code") # Google uses 'code'

    # Avoid processing code if already logged in
    if auth_code and not (creds and creds.valid):
        logging.info(f"Received Google OAuth code: {auth_code[:10]}...") # Log truncated code
        try:
            st.info("Authorization code received, exchanging for tokens...")
            # Ensure fetch_token gets the full URL including params for validation? Usually not needed.
            flow.fetch_token(code=auth_code)
            creds = flow.credentials
            st.session_state.google_creds = creds
            logging.info("Google Drive token fetched successfully.")

            # Clear the code from query params to prevent reuse on refresh/accidental clicks
            st.query_params.clear()

            # Build and store the service immediately after getting creds
            service = build('drive', 'v3', credentials=creds)
            st.session_state.google_service = service
            logging.info("Google Drive service built successfully.")
            st.success("Google Drive connection successful!")
            # Don't rerun here, let the main app logic continue and redraw
            return service, None

        except Exception as e:
            error_msg = f"Failed to fetch Google token or build service: {e}"
            st.error(error_msg)
            logging.error(error_msg, exc_info=True)
            st.query_params.clear() # Clear invalid code
            # Clear potentially partially set state
            st.session_state.google_creds = None
            st.session_state.google_service = None
            return None, f"Token exchange/service build failed: {e}"

    # If not logged in (no valid creds) and no auth code is being processed, provide the auth link
    elif not (creds and creds.valid):
        try:
            auth_url, _ = flow.authorization_url(
                access_type='offline', # Request refresh token for long-term access
                # include_granted_scopes='true', # Optional: indicates already granted scopes
                prompt='consent'      # Force consent screen to ensure refresh token
            )
            logging.info(f"Generated Google Auth URL: {auth_url}")
            # Use st.link_button for a clickable authorization link
            st.link_button("Connect Google Drive", auth_url, help="Click to authorize this app with Google Drive. You'll be redirected back here.", type="primary")
            st.info("Click the button above to authorize access to Google Drive.")
            return None, "Authentication required." # Return specific message

        except Exception as auth_url_err:
             error_msg = f"Failed to generate Google authorization URL: {auth_url_err}"
             logging.error(error_msg, exc_info=True)
             st.error(error_msg)
             return None, "Error generating auth URL."

    # Should not happen frequently if logic above is correct, but as catch-all
    elif creds and creds.valid and service:
        logging.debug("Returning existing valid service found later in the flow.")
        return service, None # Already connected
    else:
        logging.warning("Reached unexpected state in get_google_drive_service.")
        return None, "Unknown authentication state."


# --- list_google_drive_files --- (No changes needed from previous version)
def list_google_drive_files(service, page_size=100):
    """Lists files (non-folders) from the user's Google Drive."""
    if not service:
        return None, "Google Drive service not available."
    try:
        results = service.files().list(
            pageSize=page_size,
            # Consider adding `q="trashed=false"` to exclude trashed files
            q="mimeType != 'application/vnd.google-apps.folder' and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)").execute()
        items = results.get('files', [])
        # Explicit filtering might still be needed if 'q' is not perfect
        files_only = [item for item in items if 'folder' not in item.get('mimeType', '').lower()]
        return files_only, None
    except HttpError as error:
        logging.error(f"An error occurred listing Google Drive files: {error}", exc_info=True)
        error_reason = getattr(error, 'reason', str(error))
        error_details = "(No further details)"
        try:
            content = getattr(error, 'content', b'').decode('utf-8', errors='ignore')
            if content:
                error_details = content
        except Exception:
            pass
        return None, f"Error listing files: {error_reason} - {error_details}"
    except Exception as e:
        logging.error(f"Unexpected error listing Google Drive files: {e}", exc_info=True)
        return None, f"Unexpected error listing files: {e}"


# --- download_google_drive_file --- (No changes needed from previous version)
def download_google_drive_file(service, file_id, file_name):
    """Downloads a file from Google Drive into memory."""
    if not service:
        return None, "Google Drive service not available."
    try:
        logging.info(f"Requesting download for Google Drive file ID: {file_id}, Name: {file_name}")
        request = service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = googleapiclient.http.MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            if status:
                logging.info(f"GDrive Download {file_id}: {int(status.progress() * 100)}%.")
        fh.seek(0)
        fh.name = file_name # Assign name for compatibility with loaders
        logging.info(f"Successfully downloaded {file_name} ({file_id}) into memory.")
        return fh, None
    except HttpError as error:
        logging.error(f"An error occurred downloading Google Drive file {file_id}: {error}", exc_info=True)
        error_reason = getattr(error, 'reason', str(error))
        error_details = "(No further details)"
        try:
            content = getattr(error, 'content', b'').decode('utf-8', errors='ignore')
            if content:
                error_details = content
        except Exception:
            pass
        return None, f"Error downloading {file_name}: {error_reason} - {error_details}"
    except Exception as e:
        logging.error(f"Unexpected error downloading Google Drive file {file_id}: {e}", exc_info=True)
        return None, f"Unexpected error downloading {file_name}: {e}"


# --- OneDrive Functions --- (Keep existing OneDrive functions as they were)
def get_ms_graph_client():
    # ... (no changes) ...
    if not MS_CLIENT_ID or MS_CLIENT_ID == "YOUR_MS_CLIENT_ID_FALLBACK":
         st.error("Microsoft Client ID is not configured. Please set the MS_CLIENT_ID in .streamlit/secrets.toml or environment variable.")
         return None
    try:
        pca = msal.PublicClientApplication(
            MS_CLIENT_ID,
            authority=MS_AUTHORITY
        )
        return pca
    except Exception as e:
        st.error(f"Failed to initialize MSAL client: {e}")
        logging.error(f"MSAL initialization failed: {e}", exc_info=True)
        return None

def acquire_ms_token_interactive(pca):
    # ... (no changes needed, device flow is independent of Google's) ...
    if not pca: return None, None, "MSAL client not initialized"
    token_cache = st.session_state.get("ms_token", msal.SerializableTokenCache())
    pca.token_cache.deserialize(token_cache.serialize()) # Load cache
    accounts = pca.get_accounts()
    result = None; error_msg = None

    if accounts:
        logging.info(f"Attempting to acquire MS token silently for account: {accounts[0]['username']}")
        result = pca.acquire_token_silent(MS_SCOPES, account=accounts[0])

    if not result:
        logging.info("No suitable MS token in cache, initiating interactive device flow.")
        try:
            flow = pca.initiate_device_flow(scopes=MS_SCOPES)
            if "user_code" not in flow:
                error_msg = "Failed to initiate MS device flow (missing user_code)."
                st.error(error_msg)
                logging.error(error_msg)
                return None, None, error_msg

            st.warning(f"To sign in to OneDrive:\n1. Open a browser to: {flow['verification_uri']}\n2. Enter the code: {flow['user_code']}")
            st.info("After signing in on the Microsoft page, click the button below.")

            # Use a session state flag to track if we are waiting for the button
            st.session_state._ms_waiting_for_signin_check = True

            if st.button("Check Microsoft Sign-in Status", key="ms_signin_check"):
                try:
                    st.info("Attempting to acquire token by device flow...")
                    result = pca.acquire_token_by_device_flow(flow) # This will block until timeout or success/failure
                    st.session_state._ms_waiting_for_signin_check = False # No longer waiting
                except Exception as e: # Catch potential timeout or other errors during acquire
                    error_msg = f"Error acquiring token via MS device flow: {e}"
                    st.error(error_msg)
                    logging.error(error_msg, exc_info=True)
                    st.session_state._ms_waiting_for_signin_check = False # No longer waiting
                    return None, None, error_msg
            else:
                # If button not pressed yet, return indicating waiting state
                return None, None, "Waiting for Microsoft sign-in confirmation."

        except Exception as init_flow_e:
             error_msg = f"Error initiating MS device flow: {init_flow_e}"
             st.error(error_msg)
             logging.error(error_msg, exc_info=True)
             st.session_state._ms_waiting_for_signin_check = False # Ensure flag is clear
             return None, None, error_msg


    # Process result (either from silent or device flow)
    if result and "access_token" in result:
        st.session_state.ms_token = pca.token_cache # Save updated cache
        st.session_state.ms_account = pca.get_accounts()[0] if pca.get_accounts() else None # Store account info
        logging.info("MS token acquired successfully.")
        # Clear waiting flag if set
        if "_ms_waiting_for_signin_check" in st.session_state:
             del st.session_state._ms_waiting_for_signin_check
        return result["access_token"], st.session_state.ms_account, None

    elif result and "error_description" in result:
        error_msg = f"MS Login Error: {result.get('error')}: {result.get('error_description')}"
        st.error(error_msg)
        logging.error(error_msg)
        # Clear waiting flag if set
        if "_ms_waiting_for_signin_check" in st.session_state:
             del st.session_state._ms_waiting_for_signin_check
        return None, None, error_msg

    # Only show generic error if not waiting for button or specific error occurred
    elif not st.session_state.get("_ms_waiting_for_signin_check"):
        error_msg = "Could not acquire MS token."
        # Don't spam error if we just showed the check button
        # st.error(error_msg) # Maybe too noisy
        logging.error(error_msg)
        return None, None, error_msg
    else:
         # Still waiting for button press
         return None, None, "Waiting for Microsoft sign-in confirmation."


def list_onedrive_files(access_token, page_size=100):
    # ... (no changes) ...
    if not access_token: return None, "OneDrive access token not available."
    graph_url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
    headers = {'Authorization': 'Bearer ' + access_token}
    # Select only necessary fields and filter out folders server-side if possible
    params = {
        '$top': page_size,
        '$select': 'id,name,file', # Request id, name, and file facet (to filter)
        # '$filter': "file ne null" # Filter for items that have a 'file' facet (are not folders)
        # Filter might not work on root children easily, manual filter is safer
    }
    try:
        response = requests.get(graph_url, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get('value', [])
        # Filter out folders (items that do *not* have a 'file' facet)
        files_only = [item for item in items if 'file' in item]
        files_info = [{'id': item['id'], 'name': item['name']} for item in files_only]
        return files_info, None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error listing OneDrive files: {e}", exc_info=True)
        error_detail = str(e)
        try:
            if e.response is not None:
                error_detail = e.response.json()['error']['message']
        except: pass
        return None, f"Error listing files: {error_detail}"
    except Exception as e:
        logging.error(f"Unexpected error listing OneDrive files: {e}", exc_info=True)
        return None, f"Unexpected error listing files: {e}"


def download_onedrive_file(access_token, file_id, file_name):
    # ... (no changes) ...
    if not access_token: return None, "OneDrive access token not available."
    download_url_endpoint = f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content"
    headers = {'Authorization': 'Bearer ' + access_token}
    try:
        logging.info(f"Requesting download for OneDrive file ID: {file_id}, Name: {file_name}")
        response = requests.get(download_url_endpoint, headers=headers, stream=True)
        response.raise_for_status()
        fh = BytesIO()
        total_downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            total_downloaded += len(chunk)
            # Add periodic logging for large files?
            # if total_downloaded % (1024 * 1024 * 5) == 0: # Log every 5MB
            #     logging.info(f"OneDrive Download {file_id}: {total_downloaded // (1024*1024)} MB...")

        fh.seek(0)
        fh.name = file_name
        logging.info(f"Successfully downloaded OneDrive file {file_name} ({file_id}), size: {total_downloaded} bytes.")
        return fh, None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading OneDrive file {file_id}: {e}", exc_info=True)
        error_detail = str(e)
        try:
            if e.response is not None:
                 error_detail = e.response.json()['error']['message']
        except: pass
        return None, f"Error downloading {file_name}: {error_detail}"
    except Exception as e:
        logging.error(f"Unexpected error downloading OneDrive file {file_id}: {e}", exc_info=True)
        return None, f"Unexpected error downloading {file_name}: {e}"