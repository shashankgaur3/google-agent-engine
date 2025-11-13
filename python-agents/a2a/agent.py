import dataiku
import logging
from dataiku.llm.python import BaseLLM

from googleagentengine.utils import (
    get_credentials_from_vertexai_connection,
    get_vertexai_agent_card,
    get_standard_a2a_agent_card,
    create_a2a_client_from_card,
    query_a2a_agent
)

# Create logger
logger = logging.getLogger("A2AAgent")


async def _inference_vertexai_a2a_agent(connection_name, reasoning_engine_resource_name, prompt, logger):
    """Inference A2A agent deployed on Vertex AI.

    Args:
        connection_name: Name of the Dataiku Vertex AI connection
        reasoning_engine_resource_name: Vertex AI Reasoning Engine resource name
            (e.g., projects/PROJECT/locations/LOCATION/reasoningEngines/ID)
        prompt: The user's query text
        logger: Logger instance

    Returns:
        Tuple of (full_response, response_text)
    """
    logger.info("Using Vertex AI A2A Agent")

    # Get the connection
    client = dataiku.api_client()
    connection = client.get_connection(connection_name)
    connection_info = connection.get_info()

    # Get credentials
    gcp_credentials = get_credentials_from_vertexai_connection(connection_info)

    # Get auth token with proper scope for Vertex AI
    from google.auth.transport.requests import Request

    # Add the required scope for Vertex AI if not already present
    if hasattr(gcp_credentials, 'scopes') and gcp_credentials.scopes:
        # Credentials already have scopes
        logger.info(f"Using existing credential scopes: {gcp_credentials.scopes}")
        gcp_credentials.refresh(Request())
    else:
        # Add cloud-platform scope for Vertex AI API access
        if hasattr(gcp_credentials, 'with_scopes'):
            logger.info("Adding cloud-platform scope to credentials")
            gcp_credentials = gcp_credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
        gcp_credentials.refresh(Request())

    auth_token = gcp_credentials.token
    logger.info("Successfully obtained auth token")

    # Get the agent card from Vertex AI
    agent_card = await get_vertexai_agent_card(reasoning_engine_resource_name, auth_token)

    # Create A2A client
    a2a_client = create_a2a_client_from_card(agent_card, auth_token)

    # Query the agent
    full_response, response_text = await query_a2a_agent(a2a_client, prompt)

    return full_response, response_text


async def _inference_standard_a2a_agent(api_token, agent_base_url, prompt, logger):
    """Inference Standard A2A Server.

    Args:
        api_token: API token for authentication
        agent_base_url: Base URL of the A2A server
        prompt: The user's query text
        logger: Logger instance

    Returns:
        Tuple of (full_response, response_text)
    """
    logger.info("Using Standard A2A Server")

    # Get the agent card using A2ACardResolver
    agent_card = await get_standard_a2a_agent_card(agent_base_url, api_token)

    # Create A2A client
    a2a_client = create_a2a_client_from_card(agent_card, api_token)

    # Query the agent
    full_response, response_text = await query_a2a_agent(a2a_client, prompt)

    return full_response, response_text


class A2AAgent(BaseLLM):
    """Custom LLM implementation for A2A Agent integration in Dataiku"""

    def __init__(self):
        pass

    def set_config(self, config, plugin_config):
        self.config = config
    
    async def aprocess(self, query, settings, trace):
        """Process a query through the A2A agent asynchronously.

        Args:
            query: Dictionary containing messages and other query parameters
            settings: LLM settings from Dataiku
            trace: Trace object for logging

        Returns:
            Dictionary with the agent's response text
        """
        # Get configuration parameters from agent.json
        auth_type = self.config.get("auth_type", "vertexai").strip()

        # Extract the user's prompt from the query
        prompt = query["messages"][-1]["content"]

        logger.info("="*70)
        logger.info("A2A Agent Query")
        logger.info("="*70)
        logger.info(f"Auth Type: {auth_type}")
        logger.info(f"Prompt: {prompt}")

        try:
            # Call the appropriate inference function based on auth type
            if auth_type == "vertexai":
                # Use Vertex AI A2A Agent
                connection_name = self.config.get("vertexai_connection").strip()
                reasoning_engine_resource_name = self.config.get("reasoning_engine_id").strip()

                logger.info(f"Connection: {connection_name}")
                logger.info(f"Reasoning Engine: {reasoning_engine_resource_name}")

                full_response, response_text = await _inference_vertexai_a2a_agent(
                    connection_name, reasoning_engine_resource_name, prompt, logger
                )
            else:  # api_token (Standard A2A Server)
                # Use Standard A2A Server
                api_token = self.config.get("api_token").strip()
                agent_base_url = self.config.get("agent_base_url").strip()

                logger.info(f"Agent Base URL: {agent_base_url}")

                full_response, response_text = await _inference_standard_a2a_agent(
                    api_token, agent_base_url, prompt, logger
                )

            # Combine all response text
            final_response = "\n".join(response_text) if response_text else "No response text received"

            logger.info("="*70)
            logger.info("Agent Response")
            logger.info("="*70)
            logger.info(final_response)
            logger.info("="*70)

            return {"text": final_response}

        except Exception as e:
            error_msg = f"Error querying A2A agent: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return {"text": error_msg}
    