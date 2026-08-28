## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
   "https://s172-29-4-51p8888.lab-aws-production.deeplearning.ai/files/Lesson_4/33768_Hedging_on_the_Frontier_.pdf",
    "https://s172-29-4-51p8888.lab-aws-production.deeplearning.ai/files/Lesson_4/34048_Second_Order_Smooth_Plan.pdf",
    "https://s172-29-4-51p8888.lab-aws-production.deeplearning.ai/files/Lesson_4/34584_Foundations_of_Equivaria.pdf",
]

papers = [
    "33768_Hedging_on_the_Frontier_.pdf",
    "34048_Second_Order_Smooth_Plan.pdf",
    "34584_Foundations_of_Equivaria.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the Hedging on the frontier "
    "and then tell me about the evaluation results"
)
response = agent.query("Give me a summary of both Hedging on the frontier and Bellman Smoothing")
print(str(response))

```

### OUTPUT:

<img width="1208" height="715" alt="image" src="https://github.com/user-attachments/assets/4a0b2e05-8733-4607-933e-d208824b8bd4" />

<img width="760" height="842" alt="image" src="https://github.com/user-attachments/assets/2590b99c-81f3-4904-89f4-3e0a489a426c" />


### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    
