# utils/jira_utils.py
import streamlit as st
from jira import JIRA
from jira.exceptions import JIRAError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_jira(server_url, email, api_token):
    """Attempts to connect to Jira and returns a client object."""
    if not server_url or not email or not api_token:
        # st.error("Jira URL, Email, and API Token are required.") # Moved error display to main app
        return None, "Jira URL, Email, and API Token are required."
    try:
        # Ensure URL is properly formatted
        if not server_url.startswith(('http://', 'https://')):
             server_url = 'https://' + server_url # Default to https
        server_url = server_url.rstrip('/') # Ensure no trailing slash

        options = {'server': server_url}
        # Use max_retries=1 for faster feedback on auth failure during interactive use
        jira_client = JIRA(options=options, basic_auth=(email, api_token), max_retries=1)
        # Test connection by fetching user info (less data than projects)
        jira_client.myself()
        logging.info(f"Successfully connected to Jira at {server_url}.")
        return jira_client, None # Return client and no error
    except JIRAError as e:
        error_message = f"Jira connection failed: Status {e.status_code} - {e.text}. Check URL, Email, Token, and Permissions."
        logging.error(f"Jira connection error: {e.status_code} - {e.text}", exc_info=False)
        return None, error_message
    except Exception as e:
        error_message = f"Jira connection failed with an unexpected error: {e}"
        logging.error(f"Jira connection error: {e}", exc_info=True)
        return None, error_message

def get_jira_projects(client):
    """Fetches accessible Jira projects."""
    if not client: return []
    try:
        return client.projects()
    except Exception as e:
        st.error(f"Failed to fetch Jira projects: {e}") # Keep UI error here as it's specific
        logging.error(f"Failed to fetch Jira projects: {e}", exc_info=True)
        return []

def format_jira_data(client, project_keys):
    """Fetches details for selected Jira projects and formats them as context."""
    if not client or not project_keys: return ""
    context = ["--- Jira Context ---"]
    max_issues_per_project = 10 # Limit issues per project to keep context manageable

    try:
        for key in project_keys:
            try:
                project = client.project(key)
                context.append(f"\n## Project: {project.name} ({project.key})")
                try:
                     lead = project.lead.displayName if hasattr(project, 'lead') and project.lead else 'N/A'
                     context.append(f"   Lead: {lead}")
                except Exception as lead_err:
                     logging.warning(f"Could not fetch lead for project {key}: {lead_err}")
                     context.append("   Lead: (Could not fetch)")

                jql = f'project = "{key}" AND statusCategory != Done ORDER BY priority DESC, updated DESC'
                issues = client.search_issues(jql, maxResults=max_issues_per_project, fields="summary,status,priority,assignee,reporter,key") # Specify fields

                if not issues:
                    context.append("   - (No open issues found matching query)")
                    continue

                context.append("\n### Recent/Priority Issues:")
                for issue in issues:
                    assignee = issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned"
                    status = issue.fields.status.name
                    priority = issue.fields.priority.name if issue.fields.priority else "N/A"
                    context.append(f"   - {issue.key}: {issue.fields.summary} (Status: {status}, Priority: {priority}, Assignee: {assignee})")

                if len(issues) == max_issues_per_project and issues.total > max_issues_per_project:
                     context.append(f"   - ... (Showing {max_issues_per_project} of {issues.total} issues matching query)")

            except JIRAError as e:
                 st.error(f"Error fetching Jira data for project {key}: {e.text}") # Keep UI error
                 logging.error(f"Error fetching Jira data for project {key}: {e.text}", exc_info=False)
                 context.append(f"\n[Error fetching data for project {key}]")
            except Exception as e:
                 st.error(f"Error processing Jira data for project {key}: {e}") # Keep UI error
                 logging.error(f"Error processing Jira data for project {key}: {e}", exc_info=True)
                 context.append(f"\n[Error processing data for project {key}]")

    except Exception as e:
        st.error(f"An unexpected error occurred while processing Jira projects: {e}") # Keep UI error
        logging.error(f"Error processing Jira projects loop: {e}", exc_info=True)

    context.append("\n--- End Jira Context ---")
    return "\n".join(context)