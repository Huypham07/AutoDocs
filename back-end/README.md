# AutoDocs Backend

The backend service for AutoDocs provides AI-powered code analysis and documentation generation capabilities.

## Features

### Core Capabilities
- Multi-language code parsing using Tree-sitter
- Graph-based repository analysis and relationship extraction
- Hierarchical clustering of code components
- RAG (Retrieval-Augmented Generation) for documentation
- Real-time analysis progress tracking via WebSocket

### API Endpoints
- Repository analysis and documentation generation
- Chat interface for interactive documentation queries
- Export functionality for generated content
- Health monitoring and status endpoints

### Supported Languages
- Python (full AST analysis)
- JavaScript/TypeScript (Tree-sitter parsing)
- Java, Go, C/C++ (basic structure analysis)

## Quick Setup

### Step 1: Environment Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt

# For multi-language support
pip install -r requirements-multilang.txt
```

### Step 3: Database Setup

#### Neo4j (Graph Database)
1. Install Neo4j Desktop or run via Docker:
```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```
2. Access Neo4j Browser at http://localhost:7474

#### MongoDB (Document Storage)
1. Install MongoDB or run via Docker:
```bash
docker run -p 27017:27017 mongo:latest
```

### Step 4: Environment Configuration

Create a `.env` file in the backend directory:

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
MONGODB_URL=localhost:27017
DATABASE_NAME=autodocs

# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=autodocs

# Optional: Local LLM
OLLAMA_HOST=localhost:11434

# Server Configuration
PORT=8000
NODE_ENV=development
```

### Step 5: Start the Server

```bash
cd src
python main.py
```

The API will be available at http://localhost:8000

### Step 6: Start Background Worker (Optional)

For asynchronous document processing:

```bash
python -m application.doc_worker
```

## Development

### Project Structure
```
src/
├── api/          # FastAPI routes and models
├── application/  # Business logic and services
├── domain/       # Core domain logic and preparators
├── infra/        # Infrastructure (databases, external APIs)
├── shared/       # Shared utilities and configurations
└── evaluator/    # Quality assessment tools
```

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

```bash
# Type checking
python -m mypy src/

# Linting
python -m flake8 src/

# Formatting
python -m black src/
```

## API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **Tree-sitter installation fails**: Install build tools for your platform
2. **Neo4j connection error**: Verify Neo4j is running and credentials are correct
3. **Memory issues with large repositories**: Increase Python memory limits or use chunked processing

### Logs

Check application logs in the `logs/` directory for detailed error information.

## How It Works
