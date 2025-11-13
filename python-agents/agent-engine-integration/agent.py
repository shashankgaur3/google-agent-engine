import dataiku
import logging
from dataiku.llm.python import BaseLLM

from google.cloud.aiplatform_v1 import ReasoningEngineExecutionServiceClient, ReasoningEngineServiceClient
from google.cloud.aiplatform_v1.types import QueryReasoningEngineRequest, StreamQueryReasoningEngineRequest

from googleagentengine.utils import get_credentials_from_vertexai_connection
    

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
        gcp_credentials = get_credentials_from_vertexai_connection(connection_info)

        # Create ReasoningEngineExecutionServiceClient
        reasoning_client = ReasoningEngineExecutionServiceClient(credentials=gcp_credentials)
 
        prompt = query["messages"][-1]["content"]

        self.logger.info(f"Query: {prompt}")

        # Create the request
        request = QueryReasoningEngineRequest(
            name=agent_resource_name,
            input={"input": prompt}
        )

        # Execute the query
        response = reasoning_client.query_reasoning_engine(request=request)

        # Display the response
        self.logger.info("="*50)
        self.logger.info(f"Agent Response: {response}")
        self.logger.info("="*50)
        output_dict = dict(response.output)

        return {"text": response.output}
    