# StudioOS

StudioOS is an AI-powered venture analysis platform designed to help entrepreneurs and investors validate ideas, analyze markets, and generate strategic insights. It orchestrates a pipeline of 14 specialized AI modules, ranging from problem discovery to pitch deck generation.

## Project Structure

The project is divided into two main components:

- **`studio-os/` (Frontend):** A React application built with Vite and TypeScript. It provides the user interface for managing projects, interacting with modules, and visualizing results.
- **`backend/` (Backend):** A Python FastAPI application that powers the heavy-lifting AI modules (M2-M6) using OpenAI's GPT-5.1 and external data sources (SerpApi, Reddit, etc.).

### Key Directories

- `studio-os/`: Frontend source code.
    - `src/components/`: React components (Sidebar, ModuleView, etc.).
    - `src/services/`: API services for communicating with the backend and OpenAI.
    - `src/constants.ts`: System prompts and module definitions.
- `backend/`: Backend source code.
    - `modules/`: Individual Python modules for specific tasks (e.g., `m2_module.py` for Problem Validation).
    - `app.py`: Main FastAPI application entry point.
- `test_modules.py`: Utility script for verifying backend modules in isolation.

## Prerequisites

- **Node.js:** v18+
- **Python:** 3.10+
- **OpenAI API Key:** Required for all AI generation.
- **SerpApi Key:** Required for Module 2 (Market Research).

## Setup & Installation

### 1. Backend Setup

Navigate to the `backend` directory (or root) and set up the Python environment:

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Frontend Setup

Navigate to the `studio-os` directory:

```bash
cd studio-os
npm install
```

## Running the Application

You need to run both the backend and frontend servers.

### Start the Backend

From the project root:

```bash
# Make sure your virtual environment is active
source venv/bin/activate

# Set your API keys
export OPENAI_API_KEY="your_openai_key"
export SERPAPI_KEY="your_serpapi_key"

# Run the server
python3 backend/app.py
```
The backend will run on `http://localhost:8000`.

### Start the Frontend

From the `studio-os` directory:

```bash
cd studio-os
npm run dev
```
The frontend will run on `http://localhost:3000`.

## Features

- **Continuous Mode:** Automatically runs through the module pipeline.
- **Hybrid Architecture:** Combines direct client-side AI calls (for speed/privacy) with server-side heavy processing (for complex data gathering).
- **In-Text Citations:** AI analysis is grounded in real-world data with verifiable links.
- **PDF Export:** Generate comprehensive venture reports.

## Modules Overview

1.  **Problem Discovery:** Trend detection.
2.  **Problem Validation:** Qualitative evidence gathering (Reddit/Socials).
3.  **Cost Analysis:** Quantifying the cost of inaction.
4.  **Current Solutions:** Competitive landscape analysis.
5.  **Idea Generation:** Venture concepting.
6.  **Market Landscape:** Deep dive due diligence.
7.  **Problem-Solution Fit:** Investment rationale.
8.  **Product Outline:** MVP definition.
9.  **PRD Generation:** Detailed requirements.
10. **Roadmap:** Execution planning.
11. **Business Model:** Revenue strategy.
12. **Financial Analysis:** Market sizing & projections.
13. **Risk Assessment:** Success factors & risks.
14. **Output Generation:** Pitch deck outline.
