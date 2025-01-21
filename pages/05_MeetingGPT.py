import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
from openai import OpenAI
import os

# kill switch: whisper모델 사용 비용이 비싸므로 이미 transcript가 마련되어 있다면 모든 과정을 pass 하도록 함
has_transcript = os.path.exists("./.cache/podcast.txt")


@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    client = OpenAI()

    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
            text_file.write(transcript.text)


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    # audio_path를 새롭게 만들어내기 보다는 확장자만 변경
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk_path = f"{chunks_folder}/chunk_{i:03d}.mp3"

        directory = os.path.dirname(chunk_path)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        chunk.export(
            chunk_path,
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    with st.status("Loading video..."):
        # video file 읽어오기
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        transcript_path = video_path.replace("mp4", "txt")
        with open(video_path, "wb") as f:  # wb: write binary
            f.write(video_content)  # video file 쓰기
    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)
    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
    with st.status("Transcribing audio..."):
        transcribe_chunks(chunks_folder, transcript_path)
