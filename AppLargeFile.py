import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as gemini
from google.cloud import storage, speech_v1p1beta1 as speech
from google.oauth2 import service_account
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import os
import tempfile
import shutil
import moviepy.editor as mp
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables from .env file
load_dotenv()

#Relative Path
json_file_path = os.path.join(os.path.dirname(__file__), 'data', 'vocal-operand-433508-b7-94cabdb6cf25.json')
# Set up Google Cloud credentials
credentials = service_account.Credentials.from_service_account_file(json_file_path)
#credentials = service_account.Credentials.from_service_account_file('data/vocal-operand-433508-b7-94cabdb6cf25.json')

# Initialize Google Cloud Storage and Speech-to-Text clients
storage_client = storage.Client(credentials=credentials)
speech_client = speech.SpeechClient(credentials=credentials)

# Google Cloud Storage bucket and folder names
bucket_name =os.getenv("BUCKET_NAME")
audio_folder =os.getenv("MY_AUDIO_FLDR_NM")

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

def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    return f"gs://{bucket_name}/{destination_blob_name}"  # Ensure the correct format

def transcribe_audio_gcs(gcs_uri):
    """Transcribes audio stored in Google Cloud Storage."""
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Adjust if necessary
        language_code="en-US",
    )

    operation = speech_client.long_running_recognize(config=config, audio=audio)
    st.write("Transcription in progress...")

    # Wait for the operation to complete
    response = operation.result(timeout=600)  # Adjust timeout as necessary
    return response

def convert_video_to_audio(uploaded_file):
    """Converts video file to WAV audio file, resamples it to 16000 Hz, and converts to mono."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_video.mp4')
    audio_path = os.path.join(temp_dir, 'temp_audio.wav')

    try:
        # Save uploaded video file to a temporary path
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())

        # Convert video to audio
        video_clip = mp.VideoFileClip(temp_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video_clip.close()

        # Resample audio to 16000 Hz and convert to mono
        audio_segment = AudioSegment.from_wav(audio_path)
        audio_segment = audio_segment.set_frame_rate(16000)  # Resample to 16000 Hz
        audio_segment = audio_segment.set_channels(1)  # Convert to mono
        resampled_audio_path = os.path.join(temp_dir, 'resampled_audio.wav')
        audio_segment.export(resampled_audio_path, format='wav')

        return resampled_audio_path
    except Exception as e:
        st.error(f"Error converting video to audio: {e}")
        return None
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Streamlit UI
st.title("Video Content Analyzer with Asynchronous Transcription")
st.write("Upload a video, convert it to audio, and transcribe the audio using Google Cloud Speech-to-Text.")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file:
    st.success("Video uploaded successfully.")

    # Convert video to audio
    audio_path = convert_video_to_audio(uploaded_file)
    
    if audio_path:
        # Upload audio file to Google Cloud Storage
        gcs_blob_name = os.path.join(audio_folder, 'temp_audio.wav')
        gcs_uri = upload_to_gcs(audio_path, bucket_name, gcs_blob_name)
        st.write(f"Audio file uploaded to Google Cloud Storage: {gcs_uri}")

        # Transcribe the audio
        response = transcribe_audio_gcs(gcs_uri)

        # Process and display the transcription results
        transcript = "\n".join([result.alternatives[0].transcript for result in response.results])
        st.write("Transcription Results:")
        st.write(transcript)

        # Define context based on the transcribed text
        context = transcript

        # Step 1: Upload the transcribed text and context to Pinecone
        text_id = uploaded_file.name.split('.')[0]  # Use the file name as the text ID
        with st.spinner("Uploading text and context to Pinecone..."):
            upload_embeddings(text_id, transcript, context)
        st.success(f"Text and context from video '{uploaded_file.name}' have been uploaded to Pinecone.")

        # Step 2: Query input
        query = st.text_input("Ask a question about the video content:")

        if query:
            # Step 3: Retrieve relevant documents and generate a response
            with st.spinner("Retrieving relevant information..."):
                retrieved_docs = retrieve_relevant_documents(query)
            st.write("Retrieved Documents:", retrieved_docs)

            # Generate the response using the context
            response_text = generate_response(query, retrieved_docs)
            st.write("Response:", response_text)
