# AI Assistant from Scratch

## 🚀 Project Philosophy

This project is the **first module** of an educational course on Agentic AI. The primary goal is to build a complex, multi-agent AI system from the ground up, using only Python and specialized, single-purpose libraries.

This project was built to demonstrate two core principles:
1.  **Agentic systems are about principles, not frameworks.** We explore the fundamental concepts of memory, tool use, planning, and orchestration without relying on a monolithic framework.
2.  **All core functionality can be built from scratch.** This project shows how to implement a ReAct loop, RAG, Text-to-SQL, and a multi-agent architecture using clean, understandable Python code.

The completed project serves as a "ground truth" reference before we move on to **Module 2**, where we will rebuild the same functionality using the **LangChain** framework to compare and contrast the two approaches.

## ✨ Features

- **Core Agentic Loop:** A fully implemented ReAct-style loop (Reasoning -> Action -> Observation) for autonomous decision-making.
- **Multi-Agent Architecture:** An Orchestrator-Specialist pattern where a master agent decomposes tasks and delegates them to a team of specialized agents.
- **Extensible Tool System:** A robust system for creating and integrating tools, including:
    - 🌐 **Internet Search** (`InternetSearchTool`)
    - 📄 **Web Page Browsing** (`BrowseWebpage`)
    - 🧠 **Retrieval-Augmented Generation (RAG)** (`VectorSearchTool`) for querying a private knowledge base of scientific papers.
    - 📊 **Text-to-SQL** (`SQLQueryTool`) for querying a structured database.
- **Persistent Memory:** Conversation history is maintained using Redis, allowing for context-aware interactions.
- **Role-Based Agents:** A flexible factory system to create agents with different "personalities" (prompts) and capabilities (tools).
- **Command-Line Interface:** A clean, interactive CLI for easy interaction with the agents.

## 🏛️ Project Structure

The project is organized into clear, decoupled modules:

```
.
├── agents/             # Concrete agent implementations and role factories
├── cli/                # The command-line interface (using Click)
├── config/             # Project configuration (using Pydantic)
├── core/               # Core abstractions (BaseAgent, BaseTool, Registry)
├── data/               # Raw and processed data for RAG and SQL
├── logs/               # Application log files
├── scripts/            # Helper scripts for data preparation (downloading, indexing)
├── tests/              # Unit and integration tests
└── tools/              # Implementations of all agent tools
```

## 🛠️ Tech Stack & Key Libraries

- **Package Management:** `uv`
- **Core Logic:** Python 3.12+
- **LLM Interaction:** `openai`
- **CLI:** `click`
- **Configuration:** `pydantic-settings`
- **Memory:** `redis`
- **Tools:**
    - **Web Search:** `ddgs`
    - **Web Browsing:** `httpx`, `beautifulsoup4`
    - **RAG:** `pymupdf4llm`, `sentence-transformers`, `faiss-cpu`, `numpy`
    - **Database:** `sqlite3` (standard library)

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone git@github.com:pueraeternis/assistant-from-scratch.git
cd assistant-from-scratch
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
# or using uv
uv venv
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
# or if you have a pyproject.toml
uv pip sync
```

### 4. Configure Environment Variables
Create a `.env` file in the project root by copying the `.env.example` file. Fill in your credentials.

```.env
# .env.example
# OpenAI Configuration
OPENAI_API_KEY="your_api_key"
OPENAI_API_URL="your_api_base_url"
LLM_MODEL_NAME="your_model_name"

# Redis Configuration
REDIS_URL="redis://localhost:6379/0"
```

### 5. Prepare Data and Indexes (Crucial Step!)
The agent requires data to be prepared beforehand. Run the following scripts **in this exact order**:

```bash
# 1. Download scientific papers from arXiv
uv run python scripts/download_papers.py

# 2. Process PDFs into Markdown
uv run python scripts/process_papers.py

# 3. Build the FAISS vector index
uv run python scripts/build_index.py

# 4. Set up the SQLite database
uv run python scripts/setup_database.py
```

## ▶️ How to Run

The main entry point is the CLI. The default agent role is the `orchestrator`.

### Start an Interactive Chat
```bash
uv run python -m cli.main chat
```
This will start a session with the master `orchestrator` agent.

### Example Complex Query for the Orchestrator
```
Please create a brief report comparing the salary of our highest-paid Senior Developer with the current market average salary for the same position. Identify our employee and find the market data online.
```

### Interacting with a Specific Specialist
You can also chat directly with any of the specialized agents:
```bash
# Chat with the universal assistant
uv run python -m cli.main chat --role assistant

# Chat with the researcher (can only search the web)
uv run python -m cli.main chat --role researcher

# Chat with the database analyst (can only query the DB)
uv run python -m cli.main chat --role database_analyst
```

## 🧠 Architectural Concepts Demonstrated

- **Agentic Loop (ReAct):** The `agents/openai/agent.py` implements a `for` loop in its `chat` method that simulates the Reason-Action-Observation cycle.
- **Tool Use (Function Calling):** Agents decide which tool to use and generate a structured JSON output, which is then parsed and executed.
- **RAG from Scratch:** The `scripts/` directory and `tools/vector_search.py` show a complete, manual implementation of a Retrieval-Augmented Generation pipeline.
- **Text-to-SQL:** The `database_analyst` role demonstrates how an LLM can generate and execute SQL queries based on natural language by being provided with the database schema in its prompt.
- **Multi-Agent Systems:** The **Orchestrator-Specialist** pattern is implemented via `tools/delegate_task.py` and the various agent factories in `agents/__init__.py`. An orchestrator agent delegates tasks to other, more specialized agents.
- **Dependency Injection:** The `DelegateTaskTool` receives the `get_agent` function upon creation, a clean pattern for decoupling components.

## 🔜 Next Steps: Module 2

The journey doesn't end here! The next step is to take the principles and successes of this project and rebuild it using the **LangChain** framework. This will allow us to directly compare the "from scratch" approach with an industry-standard tool, highlighting the trade-offs in complexity, speed of development, and transparency.

Next module: https://github.com/pueraeternis/assistant-langchain.git