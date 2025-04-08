from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
import chromadb
from chromadb.utils import embedding_functions
import json
import base64
import os
from google import genai
from google.genai import types
import numpy as np
from typing import List
import time
from dotenv import load_dotenv

load_dotenv() 
# Constants
BATCH_SIZE = 1000  # Number of rows to process at once
EMBEDDING_BATCH_SIZE = 100  # Number of rows to embed at once

def generate(prompt):
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            response_text += chunk.text

    return response_text
print(generate("hey gemini say hello and i am your rag assistant"))
app = FastAPI()

# MongoDB connection with optimized settings
client = MongoClient("mongodb://localhost:27017/", maxPoolSize=50, wTimeoutMS=2500)
db = client["csv_database"]
files_collection = db["files"]

# Create indexes for faster queries
files_collection.create_index([("file_name", 1)])
files_collection.create_index([("upload_date", 1)])

chroma_client = chromadb.Client()
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create a default collection (can be user/file-specific too)
collection = chroma_client.get_or_create_collection(name="csv_rag", embedding_function=embedding_func)

def process_batch(batch_df: pd.DataFrame, file_id: str, start_idx: int) -> List[str]:
    """Process a batch of rows and return document IDs"""
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in batch_df.iterrows():
        doc = row.to_dict()
        doc_str = (
            f"This row represents a data record from the uploaded CSV.\n" +
            "\n".join(f"- {key.replace('_', ' ').capitalize()}: {value}" for key, value in doc.items())
        )        
        documents.append(doc_str)
        ids.append(f"{file_id}_{start_idx + idx}")
        metadatas.append({"file_id": file_id, "row": start_idx + idx})
    
    return documents, ids, metadatas

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the CSV file in chunks
        chunks = pd.read_csv(pd.io.common.BytesIO(await file.read()), chunksize=BATCH_SIZE)
        
        # Initialize variables
        total_rows = 0
        file_id = None
        
        # Process each chunk
        for chunk_idx, chunk_df in enumerate(chunks):
            if chunk_idx == 0:
                # First chunk - save to MongoDB and get file_id
                data = chunk_df.to_dict(orient="records")
                doc = {
                    "file_name": file.filename,
                    "content": data,
                    "upload_date": datetime.now(),
                    "total_rows": len(chunk_df)
                }
                result = files_collection.insert_one(doc)
                file_id = str(result.inserted_id)
            else:
                # Update MongoDB with additional rows
                data = chunk_df.to_dict(orient="records")
                files_collection.update_one(
                    {"_id": ObjectId(file_id)},
                    {"$push": {"content": {"$each": data}}}
                )
                files_collection.update_one(
                    {"_id": ObjectId(file_id)},
                    {"$inc": {"total_rows": len(chunk_df)}}
                )
            
            # Process the chunk in smaller batches for embeddings
            for i in range(0, len(chunk_df), EMBEDDING_BATCH_SIZE):
                batch_df = chunk_df.iloc[i:i + EMBEDDING_BATCH_SIZE]
                documents, ids, metadatas = process_batch(batch_df, file_id, total_rows + i)
                
                # Add to ChromaDB
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            total_rows += len(chunk_df)
            
            # Optional: Add progress tracking
            if chunk_idx % 10 == 0:
                print(f"Processed {total_rows} rows...")

        return {"file_id": file_id, "message": f"Upload and embedding successful. Processed {total_rows} rows."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    try:
        files = []
        for file in files_collection.find():
            files.append({
                "file_id": str(file["_id"]),
                "file_name": file["file_name"],
                "total_rows": file.get("total_rows", 0)
            })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/file/{file_id}")
async def delete_file(file_id: str):
    try:
        # Convert string ID to ObjectId
        obj_id = ObjectId(file_id)
        
        # Check if file exists
        file = files_collection.find_one({"_id": obj_id})
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete the file
        files_collection.delete_one({"_id": obj_id})
        
        # Delete from ChromaDB (in batches if needed)
        collection.delete(where={"file_id": file_id})
        
        return {"message": "File deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file/{file_id}/content")
async def get_file_content(file_id: str):
    """Retrieve the full content of a specific file from MongoDB."""
    try:
        obj_id = ObjectId(file_id)
        file_doc = files_collection.find_one({"_id": obj_id})

        if not file_doc:
            raise HTTPException(status_code=404, detail="File not found")

        content = file_doc.get("content")
        if content is None:
            # Handle cases where content might not be stored directly
            # or if the structure changed. You might need to adapt this
            # based on how you store large file content.
            raise HTTPException(status_code=404, detail="File content not found or is empty")

        # Assuming content is stored as a list of dictionaries
        return {"file_id": file_id, "content": content}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file content: {str(e)}")

@app.get("/collections")
async def list_collections_with_docs():
    try:
        collections_info = []
        collections = chroma_client.list_collections()

        for col in collections:
            collection = chroma_client.get_collection(col.name)
            # Fetching a small number of documents to get the count
            results = collection.get(include=['documents'])

            collections_info.append({
                "collection_name": col.name,
                "document_count": len(results['ids']),
                "sample_ids": results['ids'][:5]  # Optional: show first 5 IDs
            })

        return {"collections": collections_info}
    except Exception as e:
        return {"error": str(e)}

class QueryInput(BaseModel):
    query: str
    file_id: str
    top_k: int = 3

@app.post("/query")
async def query_csv(input: QueryInput):
    try:
        results = collection.query(
            query_texts=[input.query],
            n_results=input.top_k,
            where={"file_id": input.file_id}
        )

        matched_docs = results['documents'][0]

        context = "\n".join(matched_docs)
        prompt = f"""
                You are a helpful assistant. You are given a user's question and some context from a CSV file.

                Context:
                {context}

                Question: {input.query}

                Answer:"""

        answer = generate(prompt)

        return {
            "query": input.query,
            "matches": matched_docs,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)