# GraphDB Project

A Python-based graph database implementation with utilities for data retrieval and chain processing, powered by Claude Sonnet for advanced language processing capabilities.

## Project Structure

- `main.py`: Main application entry point and Claude Sonnet integration
- `chain.py`: Implementation of chain processing functionality
- `graph_utils.py`: Utility functions for graph operations
- `retriever.py`: Data retrieval and querying functionality
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Create a `.env` file in the root directory
- Add your Anthropic API key:
  ```
  ANTHROPIC_API_KEY=your_api_key_here
  ```

## Usage

Run the main application:
```bash
python main.py
```

## Features

- Graph database operations with Claude Sonnet integration
- Intelligent chain processing capabilities
- Advanced natural language understanding and processing
- Efficient data retrieval and querying
- Comprehensive graph utility functions

## Dependencies

Key dependencies include:
- anthropic: Claude Sonnet API integration
- langchain: For chain processing and LLM operations
- streamlit: Web interface
- graphviz: Graph visualization
- Other dependencies listed in `requirements.txt`

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

---
For more information or support, please [contact details or issue reporting guidelines]
