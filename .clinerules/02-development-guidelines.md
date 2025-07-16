# Development Guidelines

## Environment Setup
- Always use virtual environments for Python development
- Ensure `.env` file contains required API keys: OPENAI_API_KEY, COHERE_API_KEY
- Use Python 3.11+ for compatibility with all dependencies
- Install dependencies from requirements.txt

## Code Style for Notebooks
- Use clear, descriptive cell comments explaining each step
- Include markdown cells to explain concepts and methodology
- Follow consistent variable naming: use descriptive names like `retriever`, `embeddings`, `vector_store`
- Add error handling for API calls and external services
- Include timing information for performance analysis

## Notebook Organization
- Start each notebook with clear objectives and prerequisites
- Include imports section with all required libraries
- Add configuration section for API keys and parameters
- Structure code in logical sections: Setup → Data Loading → Processing → Evaluation
- End with summary of results and next steps

## Error Handling
- Implement graceful handling of API rate limits
- Include fallback options for different LLM providers
- Add validation for required environment variables
- Handle vector database connection issues
- Provide clear error messages for common setup problems

## Performance Considerations
- Monitor token usage and costs for LLM calls
- Implement caching for embeddings and retrieved results
- Use batch processing for large datasets
- Include memory usage monitoring for large vector operations
- Optimize chunk sizes based on content type
