import json
from google.oauth2 import service_account, credentials
from google.cloud.aiplatform_v1 import ReasoningEngineServiceClient
from google.cloud.aiplatform_v1.types import ListReasoningEnginesRequest


def _get_credentials(connection_info):
    """Get Google Cloud credentials from Dataiku connection"""

    connection_params = connection_info.get_resolved_params()

    if connection_params['authType'] == "KEYPAIR":
        if 'appSecretContent' in connection_params:
            keyRaw = connection_params['appSecretContent']
        elif 'keyPath' in connection_params:
            keyRaw = connection_params['keyPath']
        else:
            raise ValueError("No keypair found in connection. Please refer to DSS Service Account Auth documentation.")
        key = json.loads(keyRaw)
        gcp_credentials = service_account.Credentials.from_service_account_info(key)

    elif connection_params['authType'] == "OAUTH":
        if 'accessToken' not in connection_info['resolvedOAuth2Credential']:
            raise ValueError("No accessToken found in connection. Please refer to DSS OAuth2 credentials documentation.")
        accessToken = connection_info['resolvedOAuth2Credential']['accessToken']
        gcp_credentials = credentials.Credentials(accessToken)

    else:
        raise ValueError("Unsupported authentication type '%s'." % connection_params['authType'])

    return gcp_credentials


def do(payload, config, plugin_config, inputs):
    """
    Fetch the list of agents from Vertex AI Agent Engine

    Args:
        payload: Request payload
        config: Agent configuration
        plugin_config: Plugin configuration
        inputs: Input values from the form

    Returns:
        Dictionary with choices list containing agent options
    """
    import dataiku

    # Get connection and parameters
    gcp_connection_name = config.get('vertexai_connection')
    gcp_project = config.get('gcp_project')
    gcp_region = config.get('gcp_region', 'us-central1')

    # Validate required parameters
    if not gcp_connection_name:
        return {"choices": []}

    if not gcp_project:
        return {"choices": []}

    try:
        # Get GCP connection
        client = dataiku.api_client()
        connection = client.get_connection(gcp_connection_name)
        connection_info = connection.get_info()

        # Get credentials
        gcp_credentials = _get_credentials(connection_info)

        # List agents from Vertex AI Agent Engine using Reasoning Engine Service
        # Using the parent path format: projects/{project}/locations/{location}
        parent = f"projects/{gcp_project}/locations/{gcp_region}"

        # Fetch agents using the Reasoning Engine API
        agents_list = []
        try:
            # Create Reasoning Engine Service client
            reasoning_client = ReasoningEngineServiceClient(credentials=gcp_credentials)
            request = ListReasoningEnginesRequest(parent=parent)

            # List all reasoning engines (agents) in the project/region
            reasoning_engines_response = reasoning_client.list_reasoning_engines(request=request)

            for engine in reasoning_engines_response:
                agents_list.append({
                    "label": engine.display_name if hasattr(engine, 'display_name') and engine.display_name else engine.name.split('/')[-1],
                    "value": engine.name.split('/')[-1]  # Extract agent ID from full name
                })
        except Exception as e:
            # If the API fails, return a helpful error message
            return {
                "choices": [],
                "error": f"Unable to fetch agents from Vertex AI Agent Engine: {str(e)}"
            }

        if not agents_list:
            return {
                "choices": [],
                "message": "No agents found in the specified project and region"
            }

        return {"choices": agents_list}

    except Exception as e:
        # Return empty choices with error for debugging
        return {
            "choices": [],
            "error": f"Error fetching agents: {str(e)}"
        }
