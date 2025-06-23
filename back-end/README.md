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

# Server Configuration
# Optional Fields
PROJECT_NAME=AutoDocs
PORT=8000
NODE_ENV=development
```

### Step 3: Start Server

```bash
python -m api.main
```

## How It Works
