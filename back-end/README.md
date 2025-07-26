# AutoDocs Backend

This is the backend for AutoDocs, AI-powered documentation generation.

## Features

## Quick Setup

From the `backend` directory follow these steps:

### Step 1: Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Set Up Environment Variables

Create a `.env` file:

```
# Required API Keys
GOOGLE_API_KEY=
OPENAI_API_KEY=

AWS_PROFILE_NAME=
AWS_REGION_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
# Server Configuration
MONGODB_URL=
DATABASE_NAME=
RABBITMQ_URL=

# Ollama Host
OLLAMA_HOST=
```

### Step 3: Start Server

```bash
# Move to src
cd src
python main.py
```

### Step 4: Start Worker

```bash
python -m application.doc_worker
```

### Evaluate

```bash
# In src directory
python -m evaluator.evaluate
```

## How It Works
