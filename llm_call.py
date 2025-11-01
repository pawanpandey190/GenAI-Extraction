from dotenv import load_dotenv
from openai import AzureOpenAI
import os
from chromadb.utils.embedding_functions import EmbeddingFunction
load_dotenv()
# for the LLM call
def init_azure_client():
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview")
    deployment = os.getenv("AZURE_DEPLOYMENT")

    if not api_key or not endpoint:
        raise EnvironmentError(
            "Missing Azure OpenAI credentials. Ensure AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT are set."
        )

    client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    return client, deployment


# embedding call

# def Embedding_call():
#     api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_KEY")
#     endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-12-01-preview")
#     deployment = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS")
#     if not api_key or not endpoint:
#         raise EnvironmentError(
#             "Missing Azure OpenAI credentials. Ensure AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT are set."
#         )

#     client_embedd = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
#     return client_embedd, deployment


class AzureEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-12-01-preview")
        self.deployment = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS")

        if not self.api_key or not self.endpoint:
            raise EnvironmentError("Missing Azure OpenAI credentials.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )

    def __call__(self, texts):
        """This method is called by Chroma when embeddings are needed."""
        response = self.client.embeddings.create(
            model=self.deployment,
            input=texts
        )
        return [item.embedding for item in response.data]


        
