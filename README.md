# RAG CSV Query Solver

A full-stack application that allows you to upload CSV files, index their contents using a Retrieval-Augmented Generation (RAG) approach, and query the data using natural language. The system consists of a FastAPI backend and a Streamlit frontend for easy interaction.

## Features

- User-friendly Streamlit interface for interacting with your data
- Upload and process CSV files of any size with efficient batch processing
- Preview and validate CSV files before uploading
- Interactive query interface with visual results
- Automatic map visualization for geographical data
- Store structured data in MongoDB for persistence
- Create semantic search embeddings using ChromaDB and Sentence Transformers
- Query your CSV data using natural language
- Generate contextual answers powered by Google's Gemini 2.0 Flash model

## Architecture

The application consists of the following components:

1. **Streamlit Frontend**: Provides an intuitive UI for file management and data querying
2. **FastAPI Backend**: Handles HTTP requests, file uploads, and query processing
3. **MongoDB**: Stores the raw CSV data and metadata
4. **ChromaDB**: Vector database for storing and retrieving embeddings
5. **Sentence Transformers**: Creates semantic embeddings from CSV data
6. **Google Gemini API**: Generates natural language responses based on retrieved context


## LLM Implementation Notes

### Why Gemini Flash?

I chose to use Google's Gemini Flash model for this implementation due to hardware constraints. The application was developed on a machine with limited VRAM that couldn't effectively run large language models locally. Gemini Flash provides a good balance of performance and resource efficiency through its API, allowing the system to function with minimal latency even on hardware-constrained environments.

### Open Source Alternatives

For those wanting to make this application completely open source or run it locally with sufficient hardware, consider these alternatives:

1. **Mistral AI's Falcon 7B Instruct** - An excellent open source model that provides similar capabilities while being fully deployable on local hardware with around 16GB of VRAM.

2. **DeepSeek AI** - Another great open source alternative that delivers strong performance for RAG applications while allowing full control over the model deployment.

3. **Llama 2/3** - Meta's open source models provide various sizes to accommodate different hardware configurations.

### Implementation Changes

To use an open source model instead of Gemini:

1. Replace the `generate()` function in `main.py` with your chosen model's API or direct integration
2. Adjust prompt formatting as needed for your selected model
3. Update environment variable requirements accordingly

This would make the system fully open source while maintaining the core RAG functionality for CSV data querying.


## Screenshots


### File Upload and Preview
![File Upload and Preview](Screenshot_2025-04-09_054411.png)
*CSV file validation and preview before uploading*

### File Management
![File Management](Screenshot_2025-04-09_054502.png)
*Manage uploaded files with easy access to viewing and deletion*

### Data Query Interface
![Data Query Interface](Screenshot_2025-04-09_054521.png)
*Ask questions about your CSV data in natural language*

## Pydantic Models

The application uses Pydantic models for data validation and API documentation:

```python
# API Input model for querying data
class QueryInput(BaseModel):
    query: str
    file_id: str
    top_k: int = 3
```

Additional models can be added as needed for request/response validation:

```python
# Example of a File response model
class FileInfo(BaseModel):
    file_id: str
    file_name: str
    total_rows: int
    
# Example of a collection info model
class CollectionInfo(BaseModel):
    collection_name: str
    document_count: int
    sample_ids: List[str]
```

## Installation

### Prerequisites

- Python 3.8+
- MongoDB running locally or accessible
- Google API key for Gemini access

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-csv-query-solver.git
   cd rag-csv-query-solver
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   pip install fastapi uvicorn pandas pymongo chromadb sentence-transformers google-generativeai python-multipart pydantic python-dotenv
   ```

4. Install frontend dependencies:
   ```bash
   pip install streamlit streamlit-folium folium
   ```

5. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Running the Application

1. Start the backend API:
   ```bash
   # Development mode with auto-reload
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   
   # Production mode
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. Start the Streamlit frontend:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser and navigate to:
   - Backend API documentation: `http://localhost:8000/docs`
   - Streamlit frontend: `http://localhost:8501`

### Streamlit Frontend Features

The Streamlit frontend provides an intuitive interface for:

1. **File Management**:
   - Upload CSV files with validation and preview
   - View a list of uploaded files with delete options
   - View full table contents of any file

2. **Data Querying**:
   - Ask natural language questions about your data
   - Adjust the number of results to retrieve
   - View answers generated by the Gemini model
   - Explore matching data points used to generate the answer

3. **Visualizations**:
   - Automatic map generation for geographical data
   - Data previews and explorations

### API Endpoints

- `POST /upload`: Upload a CSV file for processing
- `GET /files`: List all uploaded files
- `DELETE /file/{file_id}`: Delete a file and its embeddings
- `GET /file/{file_id}/content`: View the content of an uploaded file
- `GET /collections`: List all vector collections and their document counts
- `POST /query`: Query a file with natural language

### Example API Workflow

1. **Upload a CSV file**:
   ```bash
   curl -X POST -F "file=@path/to/your/file.csv" http://localhost:8000/upload
   ```

2. **List uploaded files**:
   ```bash
   curl http://localhost:8000/files
   ```

3. **Query a file**:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"query": "What is the highest value?", "file_id": "your_file_id", "top_k": 5}' http://localhost:8000/query
   ```

## Performance Optimizations

### Batch Processing Implementation

This application implements several batch processing techniques to handle large CSV files efficiently:

1. **Chunked CSV Reading**: Uses pandas' `chunksize` parameter to read large CSV files in manageable chunks
2. **Optimized MongoDB Operations**: Uses bulk operations and indexes for faster database performance
3. **Embedding Batching**: Processes embeddings in smaller batches (configurable via `EMBEDDING_BATCH_SIZE`)
4. **MongoDB Connection Pooling**: Configures connection pool and timeout settings for better performance
5. **Progressive UI Feedback**: Streamlit progress bars and spinners to keep users informed during lengthy operations

These optimizations ensure that even large CSV files (100MB+) can be processed efficiently without consuming excessive memory or causing timeouts.

### Frontend Optimizations

The Streamlit frontend includes several optimizations:

1. **CSV Validation**: Multiple fallback methods to handle different CSV formats and encodings
2. **Progressive Loading**: Load data in chunks with visual feedback
3. **Lazy Loading**: Only load full table data when explicitly requested
4. **Responsive Layout**: Adapts to different screen sizes with Streamlit's responsive design

## Common Errors and Solutions

### MongoDB Connection Issues

**Error**: `pymongo.errors.ServerSelectionTimeoutError`

**Solution**: 
- Ensure MongoDB is running (`mongod --dbpath /path/to/data/directory`)
- Check if the MongoDB connection string is correct
- Verify network connectivity if using a remote MongoDB instance

### Memory Issues with Large Files

**Error**: `MemoryError` or slow performance when processing large files

**Solution**:
- Decrease the `BATCH_SIZE` constant in the code
- Ensure your system has sufficient RAM
- Consider increasing swap space on your server

### ChromaDB Embedding Errors

**Error**: `ValueError: Embedding function error`

**Solution**:
- Check if the sentence-transformer model is correctly installed
- Verify the input data doesn't contain incompatible characters
- Try a different embedding model like "paraphrase-MiniLM-L3-v2" which is smaller

### Google API Key Issues

**Error**: `google.api_core.exceptions.InvalidArgument`

**Solution**:
- Verify your API key is correct in the `.env` file
- Check if your Google API key has the necessary permissions
- Ensure you have quota available for the Gemini model

### FastAPI Validation Errors

**Error**: `pydantic.error_wrappers.ValidationError`

**Solution**:
- Check the request payload matches the expected Pydantic model
- Ensure required fields are provided and have the correct data types
- Use the `/docs` endpoint to see the expected request format

### Streamlit Connection Errors

**Error**: `ConnectionError: Connection refused`

**Solution**:
- Make sure the FastAPI backend is running
- Check that the `API_BASE_URL` in streamlit_app.py points to the correct address
- Verify network settings allow connections between Streamlit and FastAPI

## Configuration Options

### Backend Configuration

You can customize various aspects of the backend by modifying the constants at the top of the main.py file:

```python
# Processing constants
BATCH_SIZE = 1000         # Rows per batch when reading CSV
EMBEDDING_BATCH_SIZE = 100  # Rows per batch for embedding generation

# MongoDB settings
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "csv_database"
MONGO_COLLECTION = "files"

# ChromaDB settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Frontend Configuration

The Streamlit frontend can be configured by modifying the constants in streamlit_app.py:

```python
# API connection
API_BASE_URL = "http://localhost:8000"

# UI settings
st.set_page_config(
    page_title="CSV Query System",
    page_icon="📊",
    layout="wide"
)
```

## Future Improvements

- Add user authentication and file ownership
- Implement more sophisticated data preprocessing options
- Add support for different file formats (Excel, JSON, etc.)
- Enhance the map visualization with additional options
- Add chart generation for numerical data
- Add progress tracking for large file uploads

## License

[MIT License](LICENSE)
