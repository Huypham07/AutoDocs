# AutoDocs

AI-powered documentation generation for software repositories using advanced code analysis and graph-based knowledge extraction.

## Features

### Code Analysis & Understanding

- Multi-language code parsing (Python, JavaScript, TypeScript, Java, Go, C/C++)
- Dependency graph construction and relationship mapping
- Architectural pattern detection and classification
- Code complexity analysis and quality metrics

### Graph-based Knowledge Extraction

- Multi-modal graph representation of code structure
- Hierarchical clustering of related components
- Semantic similarity analysis between modules
- Architectural layer and domain classification

### Intelligent Documentation Generation

- Context-aware documentation using Large Language Models
- RAG (Retrieval-Augmented Generation) for accurate content
- Architecture-focused documentation templates
- Interactive chat interface for documentation queries

### Repository Analysis

- GitHub integration with access token support
- Export capabilities for generated documentation
- Quality assessment and validation metrics

## Quick Start

### Prerequisites

- FastAPI for backend
- Next.js for frontend
- Neo4j database for graph storage
- MongoDB for document storage

### Clone the repository

```bash
git clone https://github.com/Huypham07/AutoDocs.git
cd AutoDocs
```

### Step 1: Start the Backend

👉 See detailed setup instructions: [Backend README](back-end/README.md)

### Step 2: Start the Frontend

👉 See detailed setup instructions: [Frontend README](front-end/README.md)

### Step 3: Using AutoDocs

1. Open http://localhost:3000 in your browser
2. Enter a GitHub repository URL
3. For private repositories, add your GitHub access token
4. Click "Analyze Repository" to start the analysis
5. View generated documentation and explore the results

## Architecture

AutoDocs uses a multi-stage pipeline for documentation generation:

1. **Code Analysis**: Parse repository structure and extract code components
2. **Graph Construction**: Build dependency graphs and relationship mappings
3. **Knowledge Extraction**: Apply clustering and semantic analysis
4. **Documentation Generation**: Use LLMs with RAG for content creation

## Technology Stack

- **Backend**: Python, FastAPI, Neo4j, MongoDB
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **AI/ML**: OpenAI GPT, Google Gemini, Ollama
- **Code Analysis**: Tree-sitter, AST parsing
- **Graph Processing**: NetworkX, community detection algorithms

## Docker Setup (Recommended)

For a quick setup using Docker:

```bash
# Start essential services (Neo4j, MongoDB, Ollama)
docker-compose up -d

# To include all services including RabbitMQ
docker-compose --profile full up -d

# View running services
docker-compose ps

# View logs
docker-compose logs -f
```

This will start:

- **Neo4j** on port 7687 (browser: http://localhost:7474, user: neo4j, password: password)
- **MongoDB** on port 27017 (admin UI: http://localhost:8081)
- **Ollama** on port 11434 (local LLM server)
- **RabbitMQ** on port 5672 (management: http://localhost:15672, optional)

## Manual Setup

If you prefer manual installation:

1. Install Neo4j Desktop or server
2. Install MongoDB Community Server
3. (Optional) Install Ollama for local LLM support
4. Follow the backend and frontend README files for detailed setup
