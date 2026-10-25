import os
import yt_dlp
import imageio_ffmpeg
from pydub import AudioSegment

DOWNLOAD_DIR = os.path.abspath("downloads")
CHUNK_DIR = os.path.join(DOWNLOAD_DIR, "chunks")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = FFMPEG_PATH


def download_youtube_audio(url: str) -> str:
    """Download a YouTube audio track and convert it to WAV."""
    if not url or not url.strip():
        raise ValueError("YouTube URL is required.")

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "noplaylist": True,
        "ffmpeg_location": FFMPEG_PATH,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise ValueError(f"No metadata returned for URL: {url}")

        base_name = ydl.prepare_filename(info)
        wav_path = os.path.splitext(base_name)[0] + ".wav"

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Downloaded WAV file was not found: {wav_path}")

    return wav_path


def convert_to_wav(input_path: str) -> str:
    """Convert input audio/video to mono 16 kHz WAV."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_path = os.path.abspath(input_path)
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10, output_dir: str = CHUNK_DIR) -> list[str]:
    """Split a WAV file into chunks and return the generated file paths."""
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_wav(wav_path)
    audio = audio.set_channels(1).set_frame_rate(16000)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i + 1:02d}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    if not chunks:
        raise ValueError("The audio file is empty.")

    return chunks


def process_input(source: str) -> list[str]:
    """Download or convert the source, then split it into audio chunks."""
    source = (source or "").strip()
    if not source:
        raise ValueError("No input source was provided.")

    if source.startswith(("http://", "https://")):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks