import dataiku
import json
from dataiku.llm.python import BaseLLM
from dataikuapi.dss.llm import DSSLLMStreamedCompletionChunk, DSSLLMStreamedCompletionFooter

from google.cloud.aiplatform_v1 import ReasoningEngineExecutionServiceClient, ReasoningEngineServiceClient
from google.cloud.aiplatform_v1.types import QueryReasoningEngineRequest, StreamQueryReasoningEngineRequest

try:
    from google.oauth2 import service_account, credentials
except ImportError as e:
    raise Exception("Unable to import google libraries. Make sure you are using a code-env where google is installed. Cause: " + str(e))


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
    

class MyLLM(BaseLLM):
    def __init__(self):
        pass

    def process(self, query, settings, trace):
        
        connection_name = self.config.get("vertexai_connection").strip()
        agent_resource_name = self.config.get("agent_id").strip()

        # Get the connection
        client = dataiku.api_client()
        connection = client.get_connection(connection_name)
        connection_info = connection.get_info()

        # Get credentials
        gcp_credentials = _get_credentials(connection_info)

        # Create ReasoningEngineExecutionServiceClient
        reasoning_client = ReasoningEngineExecutionServiceClient(credentials=gcp_credentials)
 
        prompt = query["messages"][-1]["content"]

        print(f"\nQuery: {prompt}")

        # Create the request
        request = QueryReasoningEngineRequest(
            name=agent_resource_name,
            input={"input": prompt}
        )

        # Execute the query
        response = reasoning_client.query_reasoning_engine(request=request)

        # Display the response
        print("\n" + "="*50)
        print(f"Agent Response: {response}")
        print("="*50)
        output_dict = dict(response.output)

        return {"text": response.output}
    