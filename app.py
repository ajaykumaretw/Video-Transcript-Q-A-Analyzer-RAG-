import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as gemini
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from dotenv import load_dotenv
import os
import moviepy.editor as mp
import speech_recognition as sr
import tempfile
import shutil
import time
import threading

# Suppress FutureWarning from transformers
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')

# Load environment variables from .env file
load_dotenv()
namespace = "video_transcripts"  # You can change this to any name you prefer

# Initialize API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set Gemini API key
gemini.configure(api_key=GEMINI_API_KEY)
# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define Pinecone index name
index_name = "indexvideocontentanalyzer"

# Initialize Pinecone with the API key
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index already exists
try:
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        # Create the index if it does not exist
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension of Sentence Transformer embeddings
            metric="dotproduct",
            spec=ServerlessSpec(cloud='aws', region="us-east-1"),
        )
        st.success(f"Index '{index_name}' has been created.")
    else:
        st.info(f"Index '{index_name}' already exists.")
except Exception as e:
    st.error(f"Error checking/creating index: {e}")

# Get the index
index = pc.Index(index_name)

# Function to generate embeddings from text
def get_embeddings(text):
    embeddings = model.encode(text)
    
    # Ensure all values are non-negative
    embeddings = np.abs(embeddings)
    
    # Normalize the embeddings to have a unit norm
    embeddings = embeddings / np.linalg.norm(embeddings)
    
    return embeddings.tolist()

# Function to upload embeddings to Pinecone
def upload_embeddings(text_id, text, context):
    embeddings = get_embeddings(text)
    st.write(f"Uploading embeddings for {text_id}...")  # Debug: Print text_id being uploaded
    
    # Metadata with both context and text
    metadata = {
        "context": context,
        "text": text
    }
    
    # Pinecone expects a list of tuples in the format: [(id, vector, metadata)]
    vectors = [(text_id, embeddings, metadata)]
    
    try:
        index.upsert(vectors=vectors)
        st.success(f"Embeddings for {text_id} have been uploaded to Pinecone with metadata.")
    except Exception as e:
        st.error(f"Error uploading embeddings: {e}")

# Function to retrieve relevant documents
def retrieve_relevant_documents(query):
    query_embedding = get_embeddings(query)
    
    # Ensure all values are non-negative
    query_embedding = np.abs(query_embedding)
    
    # Normalize the vector
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    try:
        result = index.query(
            vector=query_embedding.tolist(),  # Convert numpy array to list
            top_k=5,
            include_metadata=True
        )
        documents = [match['metadata'] for match in result['matches']]
        return documents
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# Function to generate a response using Gemini's API
def generate_response(question, context):
    combined_context = "\n".join([doc.get('text', '') for doc in context])  # Use the 'text' field from metadata
    try:
        # Create a GenerativeModel instance
        model = gemini.GenerativeModel("gemini-1.5-pro")
        
        # Generate the response
        response = model.generate_content(
            f"Context: {combined_context}\n\nQuestion: {question}",
            generation_config=gemini.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
                top_p=0.8,
                top_k=40
            )
        )
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generating response."

# Function to extract audio from video and transcribe it
def transcribe_video(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_video.mp4')
    audio_path = os.path.join(temp_dir, 'temp_audio.wav')

    try:
        # Save uploaded file to a temporary path
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())

        # Extract audio from video
        video_clip = mp.VideoFileClip(temp_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video_clip.close()  # Explicitly close the video clip

        # Convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        return text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""
    finally:
        # Cleanup function with retry
        def cleanup_files():
            max_retries = 5
            for _ in range(max_retries):
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    if os.path.isdir(temp_dir):
                        shutil.rmtree(temp_dir)
                    break  # If successful, exit the loop
                except Exception as e:
                    st.warning(f"Cleanup attempt failed: {e}. Retrying...")
                    time.sleep(1)  # Wait for 1 second before retrying
            else:
                st.error("Failed to clean up temporary files after multiple attempts.")

        # Run cleanup in a separate thread
        threading.Thread(target=cleanup_files).start()

# Streamlit UI
st.title("Video Content Analyzer with RAG")
st.write("Upload a video, convert it to text, and ask questions about the content.")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file:
    st.success("Video uploaded successfully.")

    # Extract and transcribe the text from the video
    with st.spinner("Processing video and transcribing audio..."):
        transcribed_text = transcribe_video(uploaded_file)

    if transcribed_text:
        st.write("Transcribed Text:")
        st.write(transcribed_text)

        # Upload embeddings to Pinecone
        text_id = "video_content"  # Use a unique ID for each upload
        upload_embeddings(text_id, transcribed_text, transcribed_text)

        # Generate and display a response
        question = st.text_input("Ask a question about the video content:")
        if question:
            with st.spinner("Generating response..."):
                context_docs = retrieve_relevant_documents(transcribed_text)
                answer = generate_response(question, context_docs)
                st.write("Answer:")
                st.write(answer)
