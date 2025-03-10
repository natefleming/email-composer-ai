from typing import (
  Any, 
  Annotated, 
  Callable, 
  Generator,
  TypedDict, 
  Sequence, 
  Literal,
  Optional,
)

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import (
  BaseMessage, 
  AIMessage, 
  SystemMessage, 
  HumanMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents.base import Document


from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool

from pydantic import BaseModel, Field

import mlflow
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)


class AgentState(ChatAgentState):
  #messages: Annotated[Sequence[BaseMessage], add_messages]
  context: Sequence[Document]



def retrieve_email_context_factory(config: ModelConfig) -> Callable[[AgentState, RunnableConfig], dict[str, str]]:

  endpoint_name: str = config.get("retriever").get("endpoint_name")
  index_name: str = config.get("retriever").get("index_name")
  columns: Sequence[str] = config.get("retriever").get("columns", [])
  search_parameters: dict[str, Any] = config.get("retriever").get("search_parameters", {})
  k: int = search_parameters.get("k", 5)
  filter: dict[str, Any] = search_parameters.get("filter", {})

  print(f"endpoint_name: {endpoint_name}")
  print(f"index_name: {index_name}")
  print(f"columns: {columns}")
  print(f"search_parameters: {search_parameters}")
  print(f"k: {k}")
  print(f"filter: {filter}")


  @mlflow.trace()
  def retrieve_email_context(state: AgentState, config: RunnableConfig) -> dict[str, str]:
      messages: Sequence[BaseMessage] = state["messages"]
      last_message: BaseMessage | dict = messages[-1]
      content: str = last_message.content if isinstance(last_message, BaseMessage) else last_message["content"]

      vector_search: VectorStore = DatabricksVectorSearch(
          endpoint=endpoint_name,
          index_name=index_name,
          columns=columns,
          client_args={},
      )

      context: Sequence[Document] = vector_search.similarity_search(
          query=content, k=k, filter=filter
      )

      return {"context": context}
    
  return retrieve_email_context


def format_context(context: Sequence[Document]) -> str:
  formatted_context: list[str] = []
  for document in context:
    document: Document
    formatted_email: str = f"""
      ###
      Sender: {document.metadata["sender"]}
      Subject: {document.metadata["subject"]}
      Body: {document.page_content}
    """
    formatted_context.append(formatted_email)
  if len(formatted_context) > 0:
    formatted_context.append("###")

  return "\n".join(formatted_context)
  
def draft_email_factory(config: ModelConfig) -> Callable[[AgentState, RunnableConfig], dict[str, str]]:

  model_name: str = config.get("agents").get("draft_email").get("model_name")
  prompt: str = config.get("agents").get("draft_email").get("prompt")

  print(f"model_name: {model_name}")
  print(f"prompt: {prompt}")

  @mlflow.trace()
  def draft_email(state: AgentState, config: RunnableConfig) -> dict[str, BaseMessage]:

    messages: Sequence[BaseMessage] = state["messages"]

    last_message: BaseMessage | dict = messages[-1]
    content: str = last_message.content if isinstance(last_message, BaseMessage) else last_message["content"]

    tone: str = config.get("configurable", {}).get("tone", "professional")
    topic: str = "engineering"
    context: str = format_context(state["context"])

    llm: BaseChatModel = ChatDatabricks(endpoint=model_name)

    prompt_template: PromptTemplate = (
      PromptTemplate.from_template(prompt)
    )
        
    chain: RunnableSequence = (
      {
          "content": lambda x: x["content"],
          "tone": lambda x: x["tone"],
          "topic": lambda x: x["topic"],
          "context": lambda x: x["context"]
      }
      | prompt_template
      | llm
    )

    response_messages: list[BaseMessage] = chain.invoke(
      {
        "content": content,
        "tone": tone,
        "topic": topic,
        "context": context
      }
    )

    return {"messages": [response_messages]}
  
  return draft_email
  

def create_graph(config: ModelConfig) -> CompiledStateGraph:

  workflow: StateGraph = StateGraph(AgentState)

  workflow.add_node("retrieve_email_context", retrieve_email_context_factory(config))
  workflow.add_node("draft_email", draft_email_factory(config))

  workflow.add_edge("retrieve_email_context", "draft_email")

  workflow.set_entry_point("retrieve_email_context")
  workflow.set_finish_point("draft_email")
  
  return workflow.compile()
  

class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}

        messages = []
        for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}

        for event in self.agent.stream(request, stream_mode="updates", config=custom_inputs):
          for node_data in event.values():
              yield from (
                  ChatAgentChunk(**{"delta": msg}) for msg in node_data.get("messages", [])
              )


graph: CompiledStateGraph = create_graph(config)
app: ChatAgent = LangGraphChatAgent(graph)

mlflow.models.set_model(app)
