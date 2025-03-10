# Databricks notebook source
# MAGIC %pip install --quiet --upgrade langgraph langchain databricks-langchain databricks-sdk databricks-agents mlflow python-dotenv
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = (
  f"langgraph=={version('langgraph')}",
  f"langchain=={version('langchain')}",
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"databricks-agents=={version('databricks-agents')}",
  f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC from typing import (
# MAGIC   Any, 
# MAGIC   Annotated, 
# MAGIC   Callable, 
# MAGIC   Generator,
# MAGIC   TypedDict, 
# MAGIC   Sequence, 
# MAGIC   Literal,
# MAGIC   Optional,
# MAGIC )
# MAGIC
# MAGIC from langchain.prompts import PromptTemplate, ChatPromptTemplate
# MAGIC from langchain_core.messages import (
# MAGIC   BaseMessage, 
# MAGIC   AIMessage, 
# MAGIC   SystemMessage, 
# MAGIC   HumanMessage,
# MAGIC )
# MAGIC from langchain_core.runnables.config import RunnableConfig
# MAGIC from langchain_core.vectorstores.base import VectorStore
# MAGIC from langchain_core.documents.base import Document
# MAGIC
# MAGIC
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.graph import StateGraph, END
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC
# MAGIC from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
# MAGIC from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
# MAGIC
# MAGIC from pydantic import BaseModel, Field
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelConfig
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC
# MAGIC
# MAGIC class AgentState(ChatAgentState):
# MAGIC   context: Sequence[Document]
# MAGIC
# MAGIC
# MAGIC def retrieve_email_context_factory(config: ModelConfig) -> Callable[[AgentState, RunnableConfig], dict[str, str]]:
# MAGIC
# MAGIC   endpoint_name: str = config.get("retriever").get("endpoint_name")
# MAGIC   index_name: str = config.get("retriever").get("index_name")
# MAGIC   columns: Sequence[str] = config.get("retriever").get("columns", [])
# MAGIC   search_parameters: dict[str, Any] = config.get("retriever").get("search_parameters", {})
# MAGIC   k: int = search_parameters.get("k", 5)
# MAGIC   filter: dict[str, Any] = search_parameters.get("filter", {})
# MAGIC
# MAGIC   print(f"endpoint_name: {endpoint_name}")
# MAGIC   print(f"index_name: {index_name}")
# MAGIC   print(f"columns: {columns}")
# MAGIC   print(f"search_parameters: {search_parameters}")
# MAGIC   print(f"k: {k}")
# MAGIC   print(f"filter: {filter}")
# MAGIC
# MAGIC
# MAGIC   @mlflow.trace()
# MAGIC   def retrieve_email_context(state: AgentState, config: RunnableConfig) -> dict[str, str]:
# MAGIC       messages: Sequence[BaseMessage] = state["messages"]
# MAGIC       last_message: BaseMessage | dict = messages[-1]
# MAGIC       content: str = last_message.content if isinstance(last_message, BaseMessage) else last_message["content"]
# MAGIC
# MAGIC       vector_search: VectorStore = DatabricksVectorSearch(
# MAGIC           endpoint=endpoint_name,
# MAGIC           index_name=index_name,
# MAGIC           columns=columns,
# MAGIC           client_args={},
# MAGIC       )
# MAGIC
# MAGIC       context: Sequence[Document] = vector_search.similarity_search(
# MAGIC           query=content, k=k, filter=filter
# MAGIC       )
# MAGIC
# MAGIC       return {"context": context}
# MAGIC     
# MAGIC   return retrieve_email_context
# MAGIC
# MAGIC
# MAGIC def format_context(context: Sequence[Document]) -> str:
# MAGIC   formatted_context: list[str] = []
# MAGIC   for document in context:
# MAGIC     document: Document
# MAGIC     formatted_email: str = f"""
# MAGIC       ####
# MAGIC       Sender: {document.metadata["sender"]}
# MAGIC       Subject: {document.metadata["subject"]}
# MAGIC       Body: {document.page_content}
# MAGIC     """
# MAGIC     formatted_context.append(formatted_email)
# MAGIC   if len(formatted_context) > 0:
# MAGIC     formatted_context.append("####")
# MAGIC
# MAGIC   return "\n".join(formatted_context)
# MAGIC
# MAGIC   
# MAGIC def draft_email_factory(config: ModelConfig) -> Callable[[AgentState, RunnableConfig], dict[str, str]]:
# MAGIC
# MAGIC   model_name: str = config.get("agents").get("draft_email").get("model_name")
# MAGIC   prompt: str = config.get("agents").get("draft_email").get("prompt")
# MAGIC
# MAGIC   print(f"model_name: {model_name}")
# MAGIC   print(f"prompt: {prompt}")
# MAGIC
# MAGIC   @mlflow.trace()
# MAGIC   def draft_email(state: AgentState, config: RunnableConfig) -> dict[str, BaseMessage]:
# MAGIC
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC
# MAGIC     last_message: BaseMessage | dict = messages[-1]
# MAGIC     content: str = last_message.content if isinstance(last_message, BaseMessage) else last_message["content"]
# MAGIC
# MAGIC     tone: str = config.get("configurable", {}).get("tone", "professional")
# MAGIC     context: str = format_context(state["context"])
# MAGIC
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint=model_name)
# MAGIC
# MAGIC     prompt_template: PromptTemplate = (
# MAGIC       PromptTemplate.from_template(prompt)
# MAGIC     )
# MAGIC         
# MAGIC     chain: RunnableSequence = (
# MAGIC       {
# MAGIC           "content": lambda x: x["content"],
# MAGIC           "tone": lambda x: x["tone"],
# MAGIC           "context": lambda x: x["context"]
# MAGIC       }
# MAGIC       | prompt_template
# MAGIC       | llm
# MAGIC     )
# MAGIC
# MAGIC     response_messages: list[BaseMessage] = chain.invoke(
# MAGIC       {
# MAGIC         "content": content,
# MAGIC         "tone": tone,
# MAGIC         "context": context
# MAGIC       }
# MAGIC     )
# MAGIC
# MAGIC     return {"messages": [response_messages]}
# MAGIC   
# MAGIC   return draft_email
# MAGIC   
# MAGIC
# MAGIC def create_graph(config: ModelConfig) -> CompiledStateGraph:
# MAGIC
# MAGIC   workflow: StateGraph = StateGraph(AgentState)
# MAGIC
# MAGIC   workflow.add_node("retrieve_email_context", retrieve_email_context_factory(config))
# MAGIC   workflow.add_node("draft_email", draft_email_factory(config))
# MAGIC
# MAGIC   workflow.add_edge("retrieve_email_context", "draft_email")
# MAGIC
# MAGIC   workflow.set_entry_point("retrieve_email_context")
# MAGIC   workflow.set_finish_point("draft_email")
# MAGIC   
# MAGIC   return workflow.compile()
# MAGIC   
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
# MAGIC           for node_data in event.values():
# MAGIC               yield from (
# MAGIC                   ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
# MAGIC               )
# MAGIC
# MAGIC
# MAGIC model_config_file: str = "model_config.yaml"
# MAGIC config: ModelConfig = ModelConfig(development_config=model_config_file)
# MAGIC
# MAGIC graph: CompiledStateGraph = create_graph(config)
# MAGIC app: ChatAgent = LangGraphChatAgent(graph)
# MAGIC
# MAGIC mlflow.models.set_model(app)

# COMMAND ----------

from agent_as_code import graph

from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# COMMAND ----------

from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("example_input")

app.predict(example_input)

# COMMAND ----------

from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("example_input")

for event in app.predict_stream(example_input):
    print(event, "-----------\n")

# COMMAND ----------

import mlflow
from agent_as_code import config #tools, LLM_ENDPOINT_NAME


from databricks_langchain import VectorSearchRetrieverTool

from langchain.tools import Tool
from mlflow.models.resources import (
    DatabricksResource, 
    DatabricksFunction, 
    DatabricksTable,
    DatabricksVectorSearchIndex,
    DatabricksServingEndpoint
)
from mlflow.models.model import ModelInfo
from mlflow.models.signature import ModelSignature
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA, CHAT_MODEL_OUTPUT_SCHEMA


draft_email_model_name: str = config.get("agents").get("draft_email").get("model_name")
index_name: str = config.get("retriever").get("index_name")

tools: Sequence[Tool] = []

resources: list[DatabricksResource] = [
    DatabricksServingEndpoint(endpoint_name=draft_email_model_name),
    DatabricksVectorSearchIndex(index_name=index_name)
]
for tool in tools:
    tool: Tool
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

signature: ModelSignature = ModelSignature(inputs=CHAT_MODEL_INPUT_SCHEMA, outputs=CHAT_MODEL_OUTPUT_SCHEMA)

with mlflow.start_run():
    model_info: ModelInfo = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent_as_code.py",
        model_config=config.to_dict(),
        pip_requirements=pip_requirements,
        resources=resources,
    )

# COMMAND ----------

from agent_as_code import config

input_data: dict[str, Any] = config.get("app").get("example_input")

mlflow.models.predict(
    model_uri=f"runs:/{model_info.run_id}/agent",
    input_data=input_data,
)

# COMMAND ----------

from mlflow.entities.model_registry.model_version import ModelVersion
from agent_as_code import config


mlflow.set_registry_uri("databricks-uc")

registered_model_name: str = config.get("app").get("registered_model_name")

registered_model_info: ModelVersion = mlflow.register_model(
    model_uri=model_info.model_uri, 
    name=registered_model_name
)

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import ServedModelInputWorkloadSize

from agent_as_code import config

endpoint_name: str = config.get("app").get("endpoint_name")

agents.deploy(
  model_name=registered_model_name, 
  model_version=registered_model_info.version, 
  scale_to_zero=True,
  environment_vars={},
  tags = {
    "type": "poc",
    "cx": "bloomin_brands"
  },
  workload_size=ServedModelInputWorkloadSize.SMALL,
  endpoint_name=endpoint_name,
)



