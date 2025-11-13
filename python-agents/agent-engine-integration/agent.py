import dataiku
import logging
from dataiku.llm.python import BaseLLM

from google.cloud.aiplatform_v1 import ReasoningEngineExecutionServiceClient, ReasoningEngineServiceClient
from google.cloud.aiplatform_v1.types import QueryReasoningEngineRequest, StreamQueryReasoningEngineRequest

from googleagentengine.utils import get_credentials_from_vertexai_connection

# Create logger
logger = logging.getLogger("VertexAIAgent")


class VertexAIAgent(BaseLLM):
    def __init__(self):
        pass

    def set_config(self, config, plugin_config):
        self.config = config

    def process(self, query, settings, trace):
        
        connection_name = self.config.get("vertexai_connection").strip()
        agent_resource_name = self.config.get("agent_id").strip()

        # Get the connection
        client = dataiku.api_client()
        connection = client.get_connection(connection_name)
        connection_info = connection.get_info()

        # Get credentials
        gcp_credentials = get_credentials_from_vertexai_connection(connection_info)

        # Create ReasoningEngineExecutionServiceClient
        reasoning_client = ReasoningEngineExecutionServiceClient(credentials=gcp_credentials)
 
        prompt = query["messages"][-1]["content"]

        logger.info(f"Query: {prompt}")

        # Create the request
        request = QueryReasoningEngineRequest(
            name=agent_resource_name,
            input={"input": prompt}
        )

        # Execute the query
        response = reasoning_client.query_reasoning_engine(request=request)

        # Display the response
        logger.info("="*50)
        logger.info(f"Agent Response: {response}")
        logger.info("="*50)

        # Parse the response structure
        # The response.output is a Struct with nested fields
        # Extract the 'output' field from the struct_value
        output_dict = dict(response.output)

        # Get the actual text response from the nested structure
        response_text = output_dict.get('output', '')

        # If response_text is still a dict/object, convert to string
        if not isinstance(response_text, str):
            response_text = str(response_text)

        return {"text": response_text}
    