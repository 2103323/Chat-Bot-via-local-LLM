# Agentic AI

Agentic AI is an autonomous artificial intelligence framework designed to make decisions and execute tasks with minimal human intervention. It leverages generative AI models to plan, route, and execute tasks through a well‐defined architecture and set of patterns.

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
  - [Agent AI Prerequisites](#agent-ai-prerequisites)
  - [Patterns in Agentic AI](#patterns-in-agentic-ai)
  - [Framework & Architecture](#framework--architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

Agentic AI provides a comprehensive framework that integrates:
- **Autonomy in Planning & Execution:** The system autonomously decomposes complex goals into manageable tasks.
- **Dynamic Adaptability:** Continuous self-assessment, tool selection, and routing allow the AI to adapt to evolving contexts.
- **Integrated Multi-Agent Systems:** Coordination among several agents enables robust distributed problem solving.
- **Enhanced User Experience:** Natural language interfaces allow intuitive interactions and proactive decision-making.

The repository includes code examples for setting up a local language model, creating document indexes from PDFs, and routing queries through specialized tools.

## Key Concepts

### Agent AI Prerequisites

Before working with the Agentic AI framework, ensure you have a good understanding of:
- Generative AI and Large Language Models (LLMs)
- Python programming (preferably in a notebook environment)
- The Llama Index framework and related tools

### Patterns in Agentic AI

Agentic AI is built on several core patterns:

- **Planning Pattern:** Decomposes complex goals into a series of sub-tasks.
- **Routing Pattern:** Directs data and tasks to the optimal processing paths.
- **Tool Use Pattern:** Leverages pre-defined tools with specific inputs to execute actions.
- **Reflection Pattern:** Continuously self-assesses to refine outcomes and avoid errors.

### Framework & Architecture

The architecture consists of several components working in harmony:
- **Goal, Input & Memory:** Define objectives and store context.
- **Orchestrator:** Acts as the central communication hub to manage tasks.
- **Planner:** Breaks down high-level goals into actionable steps.
- **Executor:** Runs the tasks based on the planner’s strategy.
- **LLM Integration:** Provides natural language processing capabilities to drive decision-making and tool usage.

## Installation

### Prerequisites

- **Python Version:** 3.10  
- **CUDA Version:** 11.8  
  > *Note:* The CUDA version is critical if you plan to leverage GPU acceleration.

### Libraries Used

This project relies on several key libraries:

- **Llama Index:**  
  Provides a comprehensive framework for document indexing and query engine creation. It includes components such as `SimpleDirectoryReader`, `SentenceSplitter`, `VectorStoreIndex`, `QueryEngineTool`, and `RouterQueryEngine` which are central to building and querying AI agents.
  
- **HuggingFace & SentenceTransformer:**  
  Used for generating embeddings. The `HuggingFaceEmbedding` from Llama Index (or alternatively, `SentenceTransformer` directly) enables efficient text embedding for document processing.
  
- **Torch (PyTorch):**  
  Utilized for deep learning functionalities, ensuring compatibility with GPU acceleration via CUDA.
  
- **Requests:**  
  A simple HTTP library for Python used to interact with external APIs, such as the local Ollama LLM.
  
- **Nest Asyncio:**  
  Patches Python’s asyncio to allow nested event loops, which is particularly useful when running the code in environments like Jupyter Notebooks.

### Setup Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/agentic-ai.git
   cd agentic-ai
   ```

2. **Set Up a Virtual Environment (optional but recommended):**

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Ensure that you have CUDA 11.8 installed on your system if GPU support is required. Then, install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   *If you need to install CUDA, follow the official [NVIDIA CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads) for your platform.*

4. **Configure Environment Variables (if necessary):**

   Set any required environment variables (e.g., for CUDA paths) in your shell or in a `.env` file.

## Usage

After installation, you can run the provided examples or integrate the framework into your own projects. For example, to create a document index from a PDF and query it:

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Initialize the sentence splitter
splitter = SentenceSplitter(chunk_size=1024)

# Read and process your PDF document
documents = SimpleDirectoryReader(input_files=["path/to/your/document.pdf"]).load_data()
nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)

# Create and run a query engine on the index
query_engine = index.as_query_engine()
response = query_engine.query("What are the key features of Agentic AI?")
print(response)
```

Additional examples include:
- Setting up a local LLM using Ollama.
- Configuring custom embeddings with HuggingFace.
- Building and routing queries using multiple tools.

## Contributing

Contributions are welcome! Please fork the repository and open pull requests for any enhancements or bug fixes. For major changes, open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.
