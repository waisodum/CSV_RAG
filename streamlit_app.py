import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import json
import folium
from streamlit_folium import folium_static
import re
import time  # Import time for progress bar simulation

# Set page config
st.set_page_config(
    page_title="CSV Query System",
    page_icon="📊",
    layout="wide"
)

# Constants
API_BASE_URL = "http://localhost:8000"

# Helper functions
def validate_csv(file):
    try:
        # First, let's check the file content
        content = file.read().decode('utf-8')
        file.seek(0)  # Reset file pointer
        
        # Check if the file is empty
        if not content.strip():
            return False, "The file is empty"
            
        # Try different CSV reading options
        try:
            # Try with default settings first
            df = pd.read_csv(file)
            file.seek(0)
        except:
            try:
                # Try with different separators
                df = pd.read_csv(file, sep=None, engine='python')
                file.seek(0)
            except:
                try:
                    # Try with different encoding
                    df = pd.read_csv(file, encoding='latin1')
                    file.seek(0)
                except Exception as e:
                    return False, f"Could not read CSV file. Error: {str(e)}"
        
        # Check if the DataFrame is empty
        if df.empty:
            return False, "The CSV file is empty"
            
        # Check if there are any columns
        if len(df.columns) == 0:
            return False, "No columns found in the CSV file"
            
        # Show some debug information
        # st.info(f"File contains {len(df.columns)} columns and {len(df)} rows")
        # st.info(f"Columns: {', '.join(df.columns)}")
            
        return True, df
    except Exception as e:
        return False, f"Invalid CSV file: {str(e)}"

def upload_file_to_api(file):
    files = {"file": file}
    response = requests.post(f"{API_BASE_URL}/upload", files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Upload failed: {response.text}")
        return None

def get_files_from_api():
    response = requests.get(f"{API_BASE_URL}/files")
    if response.status_code == 200:
        return response.json().get("files", [])
    else:
        st.error(f"Failed to fetch files: {response.text}")
        return []

def delete_file_from_api(file_id):
    response = requests.delete(f"{API_BASE_URL}/file/{file_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Delete failed: {response.text}")
        return None

def query_file(file_id, query, top_k=3):
    data = {
        "query": query,
        "file_id": file_id,
        "top_k": top_k
    }
    response = requests.post(f"{API_BASE_URL}/query", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Query failed: {response.text}")
        return None

def get_file_content_from_api(file_id):
    response = requests.get(f"{API_BASE_URL}/file/{file_id}/content")
    if response.status_code == 200:
        return response.json().get("content")
    else:
        st.error(f"Failed to fetch file content: {response.text}")
        return None

def create_map_from_data(data):
    # Create a map centered at (0,0)
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Add markers for each location
    for item in data:
        if isinstance(item, dict):
            # Look for latitude and longitude in the data
            lat = None
            lon = None
            
            # Check common column names for coordinates
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    if lat is None and -90 <= value <= 90:
                        lat = value
                    elif lon is None and -180 <= value <= 180:
                        lon = value
            
            if lat is not None and lon is not None:
                folium.Marker(
                    location=[lat, lon],
                    popup=str(item)
                ).add_to(m)
    
    return m

def extract_coordinates(text):
    # Regular expression to find coordinates in the text
    pattern = r'[-+]?\d*\.\d+|\d+'
    coordinates = re.findall(pattern, text)
    
    if len(coordinates) >= 2:
        try:
            lat = float(coordinates[0])
            lon = float(coordinates[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    return None

# Main app
st.title("📊 CSV Query System")

# Sidebar for file management
with st.sidebar:
    st.header("File Management")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        # Validate and preview the uploaded file
        is_valid, df = validate_csv(uploaded_file)
        if is_valid:
            st.subheader("File Preview")
            st.dataframe(df.head())
            
            if st.button("Upload"):
                # Reset the file pointer before upload
                uploaded_file.seek(0)
                
                with st.spinner("Uploading file..."):
                    # Simulate progress
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.05)  # Simulate time delay
                        progress_bar.progress(percent_complete + 1)
                    
                    result = upload_file_to_api(uploaded_file)
                    progress_bar.empty() # Remove progress bar after completion
                    if result:
                        st.success("File uploaded successfully!")
                        # Force a rerun to update the file list
                        st.rerun()
        else:
            st.error(df)  # df contains the error message in this case
    
    # List and delete files
    st.subheader("Uploaded Files")
    files = get_files_from_api()
    for file in files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"📄 {file['file_name']}")
        with col2:
            if st.button("🗑️", key=f"delete_{file['file_id']}"):
                delete_file_from_api(file['file_id'])
                st.rerun()

# Main content area
st.header("Query Your Data")

# File selection
if files:
    selected_file = st.selectbox(
        "Select a file to query",
        options=files,
        format_func=lambda x: x["file_name"]
    )
    
    if selected_file:
        selected_file_id = selected_file["file_id"]

        # Button to show the full table
        if st.button("Show Full Table"):
            with st.spinner("Loading table..."):
                content = get_file_content_from_api(selected_file_id)
                if content:
                    df_full = pd.DataFrame(content)
                    st.subheader(f"Full Table for: {selected_file['file_name']}")
                    st.dataframe(df_full)
                else:
                    st.warning("Could not load file content.")

        # Query input
        query = st.text_area("Enter your question about the data")
        top_k = st.slider("Number of results to show", min_value=1, max_value=10, value=3)
        
        if st.button("Get Answer"):
            if query:
                with st.spinner("Processing your query..."):
                    result = query_file(selected_file_id, query, top_k)
                    if result:
                        # Display answer
                        st.subheader("Answer")
                        st.write(result.get("answer", "No answer available"))
                        
                        # Try to create a map if the answer contains location data
                        try:
                            # Check if the answer contains coordinates
                            coords = extract_coordinates(result.get("answer", ""))
                            if coords:
                                m = folium.Map(location=coords, zoom_start=10)
                                folium.Marker(location=coords, popup=result.get("answer", "")).add_to(m)
                                st.subheader("Location Map")
                                folium_static(m)
                        except Exception as e:
                            st.warning("Could not create map visualization")
                        
                        # Display matched documents
                        st.subheader("Relevant Data")
                        matches = result.get("matches", [])
                        for i, doc in enumerate(matches, 1):
                            with st.expander(f"Match {i}"):
                                try:
                                    doc_data = json.loads(doc)
                                    st.json(doc_data)
                                    
                                    # Try to create a map for each document if it contains location data
                                    try:
                                        m = create_map_from_data([doc_data])
                                        if m.get_bounds():
                                            st.subheader("Location Data")
                                            folium_static(m)
                                    except Exception as e:
                                        pass
                                except json.JSONDecodeError:
                                    st.write(doc)
            else:
                st.warning("Please enter a query")
else:
    st.info("Upload a CSV file to get started!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")