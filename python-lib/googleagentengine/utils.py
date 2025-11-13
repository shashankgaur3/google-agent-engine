"""
Utility functions for Google Agent Engine integration with Dataiku.

This module provides shared authentication and A2A client functionality
for connecting to Google Vertex AI Agent Engine and A2A agents.
"""

import json
import httpx
import logging

try:
    from google.oauth2 import service_account, credentials
    from google.auth.transport.requests import Request
    from a2a.client import ClientConfig, ClientFactory
    from a2a.types import TransportProtocol, Message, Part, TextPart, AgentCard
    from a2a.card_resolver import A2ACardResolver
except ImportError as e:
    raise Exception(
        "Unable to import required libraries. "
        "Make sure you are using a code-env where google and a2a libraries are installed. "
        f"Cause: {str(e)}"
    )

# Create logger
logger = logging.getLogger(__name__)


def get_credentials_from_vertexai_connection(connection_info):
    """Get Google Cloud credentials from Dataiku Vertex AI connection.

    Args:
        connection_info: Connection info object from Dataiku API

    Returns:
        Google Cloud credentials object

    Raises:
        ValueError: If connection is invalid or missing required parameters
    """
    connection_params = connection_info.get_resolved_params()

    if connection_params['authType'] == "KEYPAIR":
        if 'appSecretContent' in connection_params:
            keyRaw = connection_params['appSecretContent']
        elif 'keyPath' in connection_params:
            keyRaw = connection_params['keyPath']
        else:
            raise ValueError(
                "No keypair found in connection. "
                "Please refer to DSS Service Account Auth documentation."
            )
        key = json.loads(keyRaw)
        gcp_credentials = service_account.Credentials.from_service_account_info(key)

    elif connection_params['authType'] == "OAUTH":
        if 'accessToken' not in connection_info['resolvedOAuth2Credential']:
            raise ValueError(
                "No accessToken found in connection. "
                "Please refer to DSS OAuth2 credentials documentation."
            )
        accessToken = connection_info['resolvedOAuth2Credential']['accessToken']
        gcp_credentials = credentials.Credentials(accessToken)

    else:
        raise ValueError(f"Unsupported authentication type '{connection_params['authType']}'.")

    return gcp_credentials


async def get_vertexai_agent_card(reasoning_engine_resource_name, auth_token):
    """Get agent card from Vertex AI Reasoning Engine.

    Args:
        reasoning_engine_resource_name: Full resource name of the reasoning engine
            (e.g., projects/PROJECT/locations/LOCATION/reasoningEngines/ID)
        auth_token: Authentication token (bearer token)

    Returns:
        AgentCard object

    Raises:
        ValueError: If agent card cannot be found or parsed
        httpx.HTTPError: If API request fails
    """

    # Parse the resource name to extract components
    parts = reasoning_engine_resource_name.split('/')
    project_id = parts[parts.index('projects') + 1]
    location = parts[parts.index('locations') + 1]
    reasoning_engine_id = parts[parts.index('reasoningEngines') + 1]

    # Construct the Vertex AI API endpoint (using v1beta1 for A2A support)
    api_endpoint = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/reasoningEngines/{reasoning_engine_id}"

    logger.info(f"Fetching agent card from Vertex AI: {api_endpoint}")

    # Fetch the reasoning engine metadata
    async with httpx.AsyncClient() as client:
        response = await client.get(
            api_endpoint,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        reasoning_engine_response = response.json()

    # Extract the A2A agent card from the classMethods
    agent_card_str = None
    if 'spec' in reasoning_engine_response and 'classMethods' in reasoning_engine_response['spec']:
        for method in reasoning_engine_response['spec']['classMethods']:
            if 'a2a_agent_card' in method:
                agent_card_str = method['a2a_agent_card']
                break

    if not agent_card_str:
        raise ValueError("Could not find A2A agent card in reasoning engine response")

    # Parse the agent card JSON string
    agent_card_dict = json.loads(agent_card_str)

    # Fix the URL to point to the actual deployed agent A2A endpoint, this is due to a known issue in vertex AI SDK
    correct_url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/reasoningEngines/{reasoning_engine_id}/a2a"
    agent_card_dict['url'] = correct_url

    # Convert dict to AgentCard object
    agent_card = AgentCard.model_validate(agent_card_dict)

    logger.info("Agent card retrieved from Vertex AI")
    logger.info(f"Agent name: {agent_card.name}")
    logger.info(f"Agent URL: {agent_card.url}")

    return agent_card


async def get_standard_a2a_agent_card(agent_base_url, api_token):
    """Get agent card from Standard A2A Server using A2ACardResolver.

    Args:
        agent_base_url: Base URL of the A2A server (e.g., https://your-server.com)
        api_token: API token for authentication

    Returns:
        AgentCard object

    Raises:
        ValueError: If agent card cannot be resolved
        httpx.HTTPError: If API request fails
    """
    logger.info(f"Resolving agent card from: {agent_base_url}")

    # Create httpx client with authentication headers
    httpx_client = httpx.AsyncClient(
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
    )

    # Use A2ACardResolver to resolve the agent card
    resolver = A2ACardResolver(httpx_client=httpx_client)
    agent_card = await resolver.resolve(agent_base_url)

    logger.info("Agent card resolved from Standard A2A Server")
    logger.info(f"Agent name: {agent_card.name}")
    logger.info(f"Agent URL: {agent_card.url}")

    return agent_card


def create_a2a_client_from_card(agent_card, auth_token):
    """Create an A2A client from an agent card.

    Args:
        agent_card: AgentCard object
        auth_token: Authentication token (bearer token)

    Returns:
        A2A client instance
    """
    # Create the A2A client with authentication
    factory = ClientFactory(
        ClientConfig(
            supported_transports=[TransportProtocol.http_json],
            use_client_preference=True,
            httpx_client=httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                }
            ),
        )
    )

    a2a_client = factory.create(agent_card)
    logger.info("A2A client created")

    return a2a_client


async def query_a2a_agent(a2a_client, prompt, message_id="query-message-1"):
    """Query the A2A agent with a text message.

    Args:
        a2a_client: The authenticated A2A client
        prompt: The user's query text
        message_id: Unique message identifier

    Returns:
        Tuple of (full_response, response_text) where:
            - full_response: List of all response chunks from the agent
            - response_text: List of extracted text strings from the response
    """
    logger.info(f"Query: {prompt}")

    # Create a message
    message = Message(
        message_id=message_id,
        role="user",
        parts=[Part(root=TextPart(text=prompt))],
    )

    logger.info("Sending message to A2A agent...")

    # Send the message and collect response
    response_iterator = a2a_client.send_message(message)

    full_response = []
    response_text = []

    async for chunk in response_iterator:
        full_response.append(chunk)

        # Extract text from response if available
        if hasattr(chunk, 'message') and chunk.message:
            if hasattr(chunk.message, 'parts') and chunk.message.parts:
                for part in chunk.message.parts:
                    if hasattr(part.root, 'text'):
                        response_text.append(part.root.text)

    logger.info("Response received")

    return full_response, response_text
