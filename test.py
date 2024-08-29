import streamlit as st
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import openai
import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Define Pinecone index name
index_name = "indexvideocontentanalyzer"

# Initialize Pinecone with the API key
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index already exists
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    # Create the index if it does not exist
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of Sentence Transformer embeddings
        metric="dotproduct",
        spec=ServerlessSpec(cloud='aws', region="us-east-1"),
    )
    st.write(f"Index '{index_name}' has been created.")
else:
    st.write(f"Index '{index_name}' already exists.")

# Get the index
index = pc.Index(index_name)

# Load Sentence Transformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformer(model_name)

def get_embedding(text):
    """Get the vector embedding for the given text using Sentence Transformer."""
    vector = embeddings.encode(text)
    # Ensure vector is a list of floats and has correct dimension
    if len(vector) != 384:  # Change this based on the dimension of your index
        raise ValueError("Vector dimension mismatch")
    return vector.tolist()  # Convert numpy array to list

def convert_audio(input_file, output_file):
    """Convert audio to required format for speech recognition."""
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # 16-bit PCM
    audio = audio.set_channels(1)  # Mono
    audio.export(output_file, format="wav")

def recognize_speech_from_audio(audio_file):
    """Recognize speech from audio file using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            print("Audio data loaded successfully.")
            transcript = recognizer.recognize_google(audio_data)
            print(f"Transcript: {transcript}")
            return transcript
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def extract_audio_from_video(video_file):
    """Extract audio from video file."""
    try:
        video = VideoFileClip(video_file)
        print(f"Video details: {video}")
        audio_file = "temp_audio.wav"
        video.audio.write_audiofile(audio_file)
        print(f"Audio file created: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def generate_content(prompt, transcript_text, question=None):
    """Generate content using OpenAI based on the prompt and transcript."""
    try:
        response = openai.Completion.create(
            model="models/text-davinci-003",
            prompt=prompt.format(transcript_text, question) if question else prompt.format(transcript_text),
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def search_transcript(query):
    """Search for a query in the Pinecone index."""
    vector = get_embedding(query)
    try:
        search_results = index.query(queries=[vector], top_k=5)
        return search_results
    except Exception as e:
        st.error(f"Search error: {e}")
        return None

def extract_year(text):
    """Extract year from text."""
    match = re.search(r'\b(20\d{2})\b', text)
    return int(match.group(0)) if match else 0

# Streamlit UI
st.title("Video Transcript and Q&A")

# Initialize an empty dictionary for storing transcripts
transcripts = {}

video_files = st.file_uploader("Upload Video Files", type=["mp4", "avi", "mov"], accept_multiple_files=True)

if video_files:
    for video_file in video_files:
        st.video(video_file)
        if st.button(f"Process {video_file.name}"):
            st.write(f"Processing video file: {video_file.name}")
            audio_file = extract_audio_from_video(video_file)
            if audio_file:
                convert_audio(audio_file, 'converted_audio.wav')
                transcript_text = recognize_speech_from_audio('converted_audio.wav')
                if transcript_text:
                    transcripts[video_file.name] = transcript_text
                    st.success(f"Transcript for {video_file.name} processed successfully.")
                    # Add transcript to Pinecone index
                    vector = get_embedding(transcript_text)
                    index.upsert([(video_file.name, vector)])
                else:
                    st.error(f"Failed to process transcript for {video_file.name}.")
            else:
                st.error(f"Failed to extract audio from {video_file.name}.")
            # Display transcripts for debugging
            st.write("Current transcripts dictionary:", transcripts)

# Search functionality
st.subheader("Search Transcripts:")
search_query = st.text_input("Enter search query:")

if st.button("Search"):
    if search_query:
        results = search_transcript(search_query)
        if results:
            # Ensure results are in expected format
            if 'matches' in results:
                matches = results['matches']
                if matches:
                    sorted_results = sorted(matches, key=lambda doc: extract_year(doc['metadata']['text']), reverse=True)
                    if sorted_results:
                        st.write("Most relevant result:", sorted_results[0]['metadata']['text'])
                    else:
                        st.write("No relevant documents found.")
                else:
                    st.write("No matches found.")
            else:
                st.error("Unexpected format in search results.")
        else:
            st.error("No results found.")
    else:
        st.error("Please enter a search query.")

# Ask a question
st.subheader("Ask a Question:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if video_files and question:
        video_file = video_files[0]
        transcript_text = transcripts.get(video_file.name)
        if transcript_text:
            answer = generate_content("Answer the following question based on the transcript text. Transcript text: {} Question: {}", transcript_text, question)
            if answer:
                st.markdown("## Answer:")
                st.write(answer)
            else:
                st.error("Failed to generate answer.")
        else:
            st.error("No transcript available.")
    else:
        st.error("Please upload a video file and enter a question.")
