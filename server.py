"""
AI Learning Platform - FastAPI Server
Phục vụ video, PDF, HTML cho học tập với hỗ trợ dubbing extension
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import re
import mimetypes
import json
import asyncio
import io
import math
import subprocess
import tempfile
import hashlib
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict
import edge_tts
import time

# For Server-Sent Events
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="AI Learning Platform")

# CORS cho extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job tracking for progressive dubbing
dubbing_jobs: Dict[str, Dict] = {}
# Job structure:
# {
#   'job_id': {
#     'status': 'processing' | 'complete' | 'error',
#     'progress': 0-100,
#     'total_segments': int,
#     'completed_segments': int,
#     'current_chunk': int,
#     'audio_chunks': [paths],
#     'video_path': str,
#     'error': str (if error)
#   }
# }

# Đường dẫn gốc - thư mục chứa các khóa học
COURSES_BASE_DIR = Path(r"x:/youtube")
PLATFORM_DIR = Path(__file__).parent

def get_lesson_type(filename: str) -> str:
    """Xác định loại bài học từ extension"""
    ext = Path(filename).suffix.lower()
    if ext in ['.mp4', '.webm', '.mkv', '.avi']:
        return 'video'
    elif ext == '.html':
        return 'html'
    elif ext == '.pdf':
        return 'pdf'
    elif ext == '.srt':
        return 'subtitle'
    else:
        return 'other'

# Configure logging for TTS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple in-memory cache for generated audio
_audio_cache = {}
MAX_CACHE_SIZE = 100

class EdgeTTSEngine:
    def __init__(self, voice: str = "vi-VN-HoaiMyNeural"):
        self.voice = voice
        logger.info(f"🎤 TTS Engine initialized with voice: {self.voice}")

    def _get_cache_key(self, text: str, rate: str) -> str:
        """Generate cache key for audio"""
        return hashlib.md5(f"{self.voice}:{rate}:{text}".encode()).hexdigest()

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text to improve TTS quality"""
        import re
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove duplicate consecutive words (e.g., "the the" -> "the")
        words = text.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
            prev_word = word.lower()
        text = ' '.join(cleaned_words)
        
        # Normalize punctuation for better speech flow
        text = re.sub(r'\s+([,.])', r'\1', text)  # Remove space before comma/period
        text = re.sub(r'([,.])(?=[^\s])', r'\1 ', text)  # Add space after comma/period if missing
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods to single
        text = re.sub(r',{2,}', ',', text)  # Multiple commas to single
        
        return text.strip()

    async def generate_audio(self, text: str, start_time: float, end_time: float) -> bytes:
        """Generate audio for a segment, adjusting speed ONLY if audio is too long"""
        # Preprocess text for better quality
        text = self._preprocess_text(text)
        
        duration_srt = end_time - start_time
        if duration_srt <= 0:
            logger.warning(f"Invalid timing: start={start_time}, end={end_time}")
            return await self._synthesize(text, rate="+0%")

        logger.debug(f"🎵 Segment [{start_time:.2f}s - {end_time:.2f}s] ({duration_srt:.2f}s): {text[:50]}...")

        
        # Generate audio at normal speed first
        original_audio = await self._synthesize(text, rate="+0%")
        if not original_audio:
            logger.error("Failed to synthesize audio")
            return b""
        
        # Save to temp file to measure duration
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(original_audio)
                tmp_path = tmp.name
            
            duration_audio = self._get_duration(tmp_path)
            if duration_audio <= 0:
                logger.warning("Could not measure audio duration, using original")
                return original_audio

            # Only speed up if audio is longer than SRT slot
            # Never slow down - keep natural speed
            if duration_audio > duration_srt:
                # SPEED UP if audio is longer than SRT slot
                ratio = duration_audio / duration_srt
                
                # Cap speed up at 30% to prevent artifacts (even if it causes drift/overflow)
                # If ratio > 1.3, we just accept the drift rather than destroying audio quality
                safe_ratio = min(ratio, 1.3)
                
                percentage = int((safe_ratio - 1) * 100)
                if percentage > 0:
                    rate_str = f"+{percentage}%"
                    logger.info(f"⚡ Speed up: {duration_audio:.2f}s → {duration_srt:.2f}s (rate: {rate_str}, capped at +30%)")
                    
                    final_audio = await self._synthesize(text, rate=rate_str)
                    if final_audio:
                        return final_audio
                    else:
                        return original_audio
                else:
                    return original_audio
            else:
                # Audio fits or is shorter - use original (silence padding handled in concat)
                logger.debug(f"✅ Audio {duration_audio:.2f}s ≤ slot {duration_srt:.2f}s")
                return original_audio
                
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _synthesize(self, text: str, rate: str) -> bytes:
        cache_key = self._get_cache_key(text, rate)
        if cache_key in _audio_cache: return _audio_cache[cache_key]
        
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio": audio_data += chunk["data"]
            
            if not audio_data:
                logger.warning(f"Empty audio for: {text[:20]}")
                return b""

            if len(_audio_cache) >= MAX_CACHE_SIZE:
                _audio_cache.pop(next(iter(_audio_cache)))
            _audio_cache[cache_key] = audio_data
            return audio_data
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return b""

    def _get_duration(self, file_path: str) -> float:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            return 0.0
        except Exception as e:
            logger.warning(f"ffprobe error: {e}")
            return 0.0

tts_engine = EdgeTTSEngine()

def get_lesson_number(filename: str) -> int:
    """Trích xuất số bài học từ tên file"""
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return 999

def scan_course_folder(course_path: Path) -> list:
    """Scan thư mục khóa học và trả về danh sách bài học"""
    lessons = []
    seen_numbers = {}  # Track lessons by number to group related files
    
    if not course_path.exists():
        return lessons
    
    for file in sorted(course_path.iterdir()):
        if file.is_file():
            filename = file.name
            lesson_type = get_lesson_type(filename)
            lesson_num = get_lesson_number(filename)
            
            if lesson_type == 'other':
                continue
                
            # Tạo title từ filename (bỏ số đầu và extension)
            title = re.sub(r'^\d+\s*', '', file.stem)
            title = title.replace('_', ' ').replace('-', ' ').strip()
            
            # Normalize path - use forward slashes for URLs
            rel_path = str(file.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
            
            lesson = {
                'id': f"{lesson_num:03d}",
                'number': lesson_num,
                'filename': filename,
                'title': title,
                'type': lesson_type,
                'path': rel_path,
                'size': file.stat().st_size
            }
            
            # Nếu là video, tìm subtitle tương ứng
            if lesson_type == 'video':
                srt_file = file.with_suffix('.srt')
                if srt_file.exists():
                    lesson['subtitle'] = str(srt_file.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
                    lesson['has_subtitle'] = True
                else:
                    lesson['has_subtitle'] = False
                
                # Check for dubbed audio
                dub_file = file.with_name(f"{file.stem}_dubbed.mp3")
                if dub_file.exists():
                    lesson['dubbed_audio'] = str(dub_file.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
                    lesson['has_dubbed'] = True
                else:
                    lesson['has_dubbed'] = False
            
            lessons.append(lesson)
    
    # Sắp xếp theo số bài học
    lessons.sort(key=lambda x: (x['number'], x['type']))
    return lessons

@app.get("/api/courses")
async def list_courses():
    """Liệt kê tất cả các khóa học"""
    courses = []
    
    # 1. Kiểm tra chính thư mục COURSES_BASE_DIR xem có media trực tiếp không
    base_lessons = scan_course_folder(COURSES_BASE_DIR)
    base_video_count = len([l for l in base_lessons if l['type'] == 'video'])
    
    if base_video_count > 0:
        courses.append({
            'id': 'default-course',
            'name': 'Khóa học của tôi',
            'path': '.',
            'lessons_path': '.',
            'video_count': base_video_count,
            'total_lessons': len(base_lessons)
        })

    # 2. Kiểm tra các thư mục con
    for item in COURSES_BASE_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Tìm thư mục con chứa lessons
            course_dir = item
            subfolders = [f for f in item.iterdir() if f.is_dir() and not f.name.startswith('.')]
            
            # Nếu có subfolder, ưu tiên subfolder đầu tiên (giữ logic cũ cho các cấu trúc lồng nhau)
            # Tuy nhiên nếu subfolder có video thì mới dùng, không thì dùng chính thư mục đó
            if subfolders:
                test_lessons = scan_course_folder(subfolders[0])
                if len([l for l in test_lessons if l['type'] == 'video']) > 0:
                    course_dir = subfolders[0]
            
            # Đếm số lessons
            lessons = scan_course_folder(course_dir)
            video_count = len([l for l in lessons if l['type'] == 'video'])
            
            if video_count > 0:
                courses.append({
                    'id': item.name,
                    'name': item.name.replace('-', ' ').replace('_', ' '),
                    'path': str(item.relative_to(COURSES_BASE_DIR)).replace('\\', '/'),
                    'lessons_path': str(course_dir.relative_to(COURSES_BASE_DIR)).replace('\\', '/'),
                    'video_count': video_count,
                    'total_lessons': len(lessons)
                })
    
    return {'courses': courses}

@app.get("/api/lessons/{course_path:path}")
async def list_lessons(course_path: str):
    """Liệt kê bài học trong một khóa học"""
    course_dir = COURSES_BASE_DIR / course_path
    
    if not course_dir.exists():
        raise HTTPException(status_code=404, detail="Course not found")
    
    lessons = scan_course_folder(course_dir)
    
    # Nhóm theo type
    videos = [l for l in lessons if l['type'] == 'video']
    htmls = [l for l in lessons if l['type'] == 'html']
    pdfs = [l for l in lessons if l['type'] == 'pdf']
    
    return {
        'course': course_path,
        'lessons': lessons,
        'videos': videos,
        'htmls': htmls,
        'pdfs': pdfs,
        'counts': {
            'video': len(videos),
            'html': len(htmls),
            'pdf': len(pdfs)
        }
    }

@app.get("/media/{file_path:path}")
async def serve_media(file_path: str):
    """Serve video, PDF, HTML files"""
    full_path = COURSES_BASE_DIR / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Xác định mime type
    mime_type, _ = mimetypes.guess_type(str(full_path))
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    return FileResponse(
        full_path,
        media_type=mime_type,
        filename=full_path.name,
        content_disposition_type='inline'
    )

# ====== SPEECH-TO-TEXT ENDPOINTS ======

@app.get("/api/videos-without-srt")
async def list_videos_without_srt():
    """Liệt kê tất cả video chưa có file SRT"""
    videos_without_srt = []
    
    for course_dir in COURSES_BASE_DIR.iterdir():
        if not course_dir.is_dir() or course_dir.name.startswith('.'):
            continue
            
        # Check main dir and subdirs
        dirs_to_check = [course_dir]
        for subdir in course_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                dirs_to_check.append(subdir)
        
        for check_dir in dirs_to_check:
            for file in check_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi']:
                    srt_file = file.with_suffix('.srt')
                    if not srt_file.exists():
                        rel_path = str(file.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
                        videos_without_srt.append({
                            'path': rel_path,
                            'name': file.name,
                            'size': file.stat().st_size,
                            'course': course_dir.name
                        })
    
    return {'videos': videos_without_srt, 'count': len(videos_without_srt)}


class TranscribeRequest(BaseModel):
    video_path: str
    language: str = "en"

def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format 00:00:00,000"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt(segments: list) -> str:
    """Generate SRT content from segments"""
    srt_content = ""
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg['start'])
        end = format_srt_time(seg['end'])
        text = seg['text'].strip()
        srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt_content

@app.post("/api/transcribe")
async def transcribe_video(request: TranscribeRequest):
    """Transcribe video using Whisper and generate SRT file"""
    video_path = COURSES_BASE_DIR / request.video_path
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    srt_path = video_path.with_suffix('.srt')
    
    # Check if SRT already exists
    if srt_path.exists():
        return {
            'status': 'exists',
            'message': 'SRT file already exists',
            'srt_path': str(srt_path.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
        }
    
    try:
        # Try using local whisper first
        try:
            import whisper
            
            print(f"🎙️ Transcribing: {video_path.name}")
            
            # Load model (base is fast, medium is better quality)
            model = whisper.load_model("base")
            
            # Transcribe
            result = model.transcribe(
                str(video_path),
                language=request.language,
                task="transcribe",
                verbose=False
            )
            
            segments = result.get('segments', [])
            
        except ImportError:
            # Fallback: Use whisper CLI if installed
            print(f"🎙️ Using whisper CLI for: {video_path.name}")
            
            # Create temp dir for output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run whisper CLI
                cmd = [
                    "whisper",
                    str(video_path),
                    "--language", request.language,
                    "--output_format", "json",
                    "--output_dir", temp_dir,
                    "--model", "base"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode != 0:
                    raise Exception(f"Whisper failed: {result.stderr}")
                
                # Read JSON output
                json_file = Path(temp_dir) / f"{video_path.stem}.json"
                if json_file.exists():
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        segments = data.get('segments', [])
                else:
                    raise Exception("Whisper output not found")
        
        # Generate SRT
        if not segments:
            return {
                'status': 'error',
                'message': 'No speech detected in video'
            }
        
        srt_content = generate_srt(segments)
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"✅ Generated SRT: {srt_path.name} ({len(segments)} segments)")
        
        return {
            'status': 'success',
            'message': f'Generated {len(segments)} subtitles',
            'srt_path': str(srt_path.relative_to(COURSES_BASE_DIR)).replace('\\', '/'),
            'segment_count': len(segments)
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Transcription timed out (max 10 minutes)")
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe-all")
async def transcribe_all_videos():
    """Transcribe all videos without SRT (background process)"""
    result = await list_videos_without_srt()
    videos = result['videos']
    
    transcribed = []
    failed = []
    
    for video in videos[:5]:  # Limit to 5 at a time
        try:
            req = TranscribeRequest(video_path=video['path'])
            res = await transcribe_video(req)
            if res['status'] == 'success':
                transcribed.append(video['name'])
        except Exception as e:
            failed.append({'name': video['name'], 'error': str(e)})
    
    return {
        'transcribed': transcribed,
        'failed': failed,
        'remaining': len(videos) - len(transcribed) - len(failed)
    }

# ====== GEMINI AI INTEGRATION ======

import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini
GEMINI_AVAILABLE = False
gemini_model = None

try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip():
        genai.configure(api_key=api_key)
        model_names = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                test = gemini_model.generate_content("Say OK")
                GEMINI_AVAILABLE = True
                print(f"✅ Gemini AI enabled: {model_name}")
                break
            except:
                continue
except ImportError:
    print("⚠️ google-generativeai not installed")

class QuizRequest(BaseModel):
    subtitles: list  # List of subtitle texts
    language: str = "vi"  # Target language for questions
    num_questions: int = 5

class FlashcardRequest(BaseModel):
    subtitles: list  # List of subtitle texts
    language: str = "vi"
    num_cards: int = 10

@app.post("/api/generate-quiz")
async def generate_quiz(request: QuizRequest):
    """Generate quiz questions using Gemini AI"""
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini AI not available. Set GEMINI_API_KEY in .env", "questions": []}
    
    try:
        # Combine subtitles into context
        context = "\n".join([s if isinstance(s, str) else s.get('text', '') for s in request.subtitles[:50]])
        
        prompt = f"""Bạn là giáo viên tạo câu hỏi trắc nghiệm từ nội dung bài học.

Nội dung (phụ đề video):
{context}

Hãy tạo {request.num_questions} câu hỏi trắc nghiệm bằng tiếng Việt.

Format JSON (không có markdown):
[
  {{
    "question": "Câu hỏi...",
    "answers": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "correct": 0,
    "explanation": "Giải thích ngắn..."
  }}
]

Chỉ trả về JSON array, không có text khác."""

        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        questions = json.loads(text)
        return {"questions": questions, "count": len(questions)}
        
    except Exception as e:
        print(f"❌ Quiz generation error: {e}")
        return {"error": str(e), "questions": []}

@app.post("/api/generate-flashcards")
async def generate_flashcards(request: FlashcardRequest):
    """Generate flashcards using Gemini AI"""
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini AI not available. Set GEMINI_API_KEY in .env", "flashcards": []}
    
    try:
        # Combine subtitles into context
        context = "\n".join([s if isinstance(s, str) else s.get('text', '') for s in request.subtitles[:50]])
        
        prompt = f"""Bạn là giáo viên tạo flashcards để giúp học sinh ghi nhớ kiến thức.

Nội dung (phụ đề video):
{context}

Hãy tạo {request.num_cards} flashcards bằng tiếng Việt.
- Mặt trước: Câu hỏi ngắn hoặc thuật ngữ
- Mặt sau: Đáp án hoặc định nghĩa

Format JSON (không có markdown):
[
  {{
    "front": "Mặt trước - câu hỏi/thuật ngữ",
    "back": "Mặt sau - câu trả lời/định nghĩa"
  }}
]

Chỉ trả về JSON array, không có text khác."""

        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        
        # Parse JSON
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        flashcards = json.loads(text)
        return {"flashcards": flashcards, "count": len(flashcards)}
        
    except Exception as e:
        print(f"❌ Flashcard generation error: {e}")
        return {"error": str(e), "flashcards": []}

@app.get("/api/gemini-status")
async def gemini_status():
    """Check Gemini AI status"""
    return {
        "available": GEMINI_AVAILABLE,
        "model": gemini_model.model_name if gemini_model else None
    }

# ====== DUBBING LOGIC ======

class DubbingRequest(BaseModel):
    video_path: str
    voice: str = "vi-VN-HoaiMyNeural"

@app.post("/api/dub")
async def dub_video(request: DubbingRequest):
    """Dub a video using existing SRT and Edge-TTS"""
    video_full_path = COURSES_BASE_DIR / request.video_path
    srt_path = video_full_path.with_suffix('.srt')
    
    if not srt_path.exists():
        raise HTTPException(status_code=404, detail="SRT file not found. Transcribe first.")
    
    try:
        # 1. Parse SRT using regex (more robust)
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex for SRT block: match number, time, then text until double newline
        pattern = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})\s+(.*?)(?=\s*\n\d+\s+\d{2}:\d{2}:\d{2}|$)', re.DOTALL)
        matches = pattern.findall(content)
        
        segments = []
        for m in matches:
            segments.append({
                'start': parse_srt_timestamp(m[1]),
                'end': parse_srt_timestamp(m[2]),
                'text': m[3].strip().replace('\n', ' ')
            })
        
        if not segments:
            raise HTTPException(status_code=400, detail="Could not parse any segments from SRT.")

        logger.info(f"🎬 Dubbing video: {request.video_path} ({len(segments)} segments)")

        # 2. Deep Translate with Gemini in chunks
        translated_segments = []
        if GEMINI_AVAILABLE:
            merged = smart_merge_subtitles(segments)
            logger.info(f"📊 Merged into {len(merged)} segments for translation")
            
            # Chunking to avoid Gemini limits
            chunk_size = 20
            for i in range(0, len(merged), chunk_size):
                chunk = merged[i:i+chunk_size]
                prompt = f"Dịch các câu sau sang tiếng Việt tự nhiên cho lồng tiếng. Giữ nguyên định dạng JSON array [{{'start': float, 'end': float, 'text': string}}]. Trả về duy nhất JSON array.\n\nNội dung: {json.dumps(chunk, ensure_ascii=False)}"
                
                try:
                    response = gemini_model.generate_content(prompt)
                    match = re.search(r'\[.*\]', response.text, re.DOTALL)
                    if match:
                        translated_chunk = json.loads(match.group())
                        translated_segments.extend(translated_chunk)
                    else:
                        logger.warning(f"Chunk {i//chunk_size} translation failed, falling back")
                        translated_segments.extend(chunk)
                except Exception as ex:
                    logger.error(f"Gemini chunk error: {ex}")
                    translated_segments.extend(chunk)
        else:
            translated_segments = segments

        # 3. Generate Audio for each segment (Parallelized)
        engine = EdgeTTSEngine(voice=request.voice)
        
        async def process_segment(idx, seg):
            try:
                if idx % 5 == 0:
                    logger.info(f"🔊 Task started for segment {idx}/{len(translated_segments)}")
                return await engine.generate_audio(seg['text'], seg['start'], seg['end'])
            except Exception as e:
                logger.error(f"Segment {idx} failed: {e}")
                return b""

        tasks = [process_segment(i, s) for i, s in enumerate(translated_segments)]
        audio_segments = await asyncio.gather(*tasks)
        
        # Filter out empty segments
        valid_segments = [(seg, audio) for seg, audio in zip(translated_segments, audio_segments) if audio and len(audio) > 0]
        
        if not valid_segments:
            raise HTTPException(status_code=500, detail="Generated audio is empty.")

        logger.info(f"📝 Creating timeline-accurate audio with sequential playback...")
        
        # Build sequential audio with precise silence padding
        temp_files = []
        file_list = []  # List of files to concat in order
        
        try:
            current_position = 0.0  # Track current position in timeline
            
            for idx, (seg, audio_data) in enumerate(valid_segments):
                segment_start = seg['start']
                segment_end = seg['end']
                segment_duration = segment_end - segment_start
                
                # Calculate EXACT silence needed to reach segment start
                silence_duration = segment_start - current_position
                
                logger.debug(f"Segment {idx}: start={segment_start:.2f}s, current_pos={current_position:.2f}s, silence_needed={silence_duration:.2f}s")
                
                if silence_duration > 0.05:  # Add silence if gap > 50ms
                    silence_file = tempfile.NamedTemporaryFile(suffix=f"_silence_{idx}.mp3", delete=False)
                    silence_file.close()
                    temp_files.append(silence_file.name)
                    
                    # Generate EXACT silence duration
                    silence_cmd = [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", f"{silence_duration:.6f}",  # Use high precision
                        "-q:a", "9", "-acodec", "libmp3lame",
                        silence_file.name
                    ]
                    result = subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
                    if result.returncode == 0:
                        file_list.append(silence_file.name)
                        current_position += silence_duration
                        logger.debug(f"✅ Added {silence_duration:.3f}s silence, now at {current_position:.3f}s")
                    else:
                        logger.error(f"Failed to generate silence: {result.stderr.decode()}")
                elif silence_duration < -0.05:  # Negative gap = overlap!
                    logger.warning(f"⚠️ Segment {idx} overlap detected: {silence_duration:.2f}s")
                
                # Verify we're at the right position
                position_error = abs(current_position - segment_start)
                if position_error > 0.1:
                    logger.warning(f"⚠️ Timeline drift at segment {idx}: expected {segment_start:.2f}s, at {current_position:.2f}s (error: {position_error:.2f}s)")
                
                # Save audio segment
                audio_file = tempfile.NamedTemporaryFile(suffix=f"_audio_{idx}.mp3", delete=False)
                audio_file.write(audio_data)
                audio_file.close()
                temp_files.append(audio_file.name)
                file_list.append(audio_file.name)
                
                # Update position to segment end
                current_position = segment_end
                
                if idx % 10 == 0:
                    logger.info(f"🎵 Seg {idx+1}/{len(valid_segments)}: [{segment_start:.2f}s - {segment_end:.2f}s]")
            
            # Create concat file
            concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='_concat.txt', delete=False, encoding='utf-8')
            for f in file_list:
                # Escape single quotes and wrap path in quotes
                escaped_path = f.replace("'", "'\\''")
                concat_file.write(f"file '{escaped_path}'\n")
            concat_file.close()
            temp_files.append(concat_file.name)
            
            logger.info(f"🔧 Concatenating {len(file_list)} audio pieces...")
            
            # Concatenate all files
            dubbed_audio_path = video_full_path.with_name(f"{video_full_path.stem}_dubbed.mp3")
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file.name,
                "-acodec", "libmp3lame", "-q:a", "2",  # Re-encode to ensure compatibility
                str(dubbed_audio_path)
            ]
            
            result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg concat failed: {result.stderr.decode()}")
                raise HTTPException(status_code=500, detail="Failed to create dubbed audio")
            
            logger.info(f"✅ Dubbed audio created with sequential playback")
        
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logger.debug(f"Cleanup error: {e}")
        
        # Generate Vietnamese SRT file with EXACT same timestamps as English SRT
        vi_srt_path = video_full_path.with_name(f"{video_full_path.stem}_vi.srt")
        srt_content = ""
        
        # Use translated_segments which already have the ORIGINAL timestamps from English SRT
        for idx, seg in enumerate(translated_segments, 1):
            start_time = format_srt_time(seg['start'])
            end_time = format_srt_time(seg['end'])
            srt_content += f"{idx}\n{start_time} --> {end_time}\n{seg['text']}\n\n"
        
        with open(vi_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        rel_audio_path = str(dubbed_audio_path.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
        rel_srt_path = str(vi_srt_path.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
        logger.info(f"✅ Dubbing complete: {rel_audio_path}")
        logger.info(f"📝 Vietnamese CC: {rel_srt_path}")
        logger.info(f"⏱️ Timeline: Audio duration matches SRT timeline exactly")

        return {
            "status": "success",
            "audio_path": rel_audio_path,
            "srt_path": rel_srt_path,
            "segments": len(translated_segments)
        }
        
    except Exception as e:
        logger.error(f"❌ Dubbing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================
# PROGRESSIVE DUBBING ENDPOINTS
# ====================

# Global dictionary to store dubbing job progress
dubbing_jobs = {}

@app.post("/api/dub-async")
async def dub_async(request: DubbingRequest, background_tasks: BackgroundTasks):
    """
    Start dubbing in background - returns immediately with job ID.
    Client can track progress via SSE endpoint.
    """
    try:
        video_full_path = COURSES_BASE_DIR / request.video_path
        srt_path = video_full_path.with_suffix('.srt')
        
        if not srt_path.exists():
            raise HTTPException(status_code=404, detail="SRT file not found. Transcribe first.")
        
        # Parse SRT immediately
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})\s+(.*?)(?=\s*\n\d+\s+\d{2}:\d{2}:\d{2}|$)', re.DOTALL)
        matches = pattern.findall(content)
        
        segments = []
        for m in matches:
            segments.append({
                'start': parse_srt_timestamp(m[1]),
                'end': parse_srt_timestamp(m[2]),
                'text': m[3].strip().replace('\n', ' ')
            })
        
        if not segments:
            raise HTTPException(status_code=400, detail="Could not parse SRT")
        
        # Create job ID
        job_id = f"dub_{hash(request.video_path)}_{int(time.time() * 1000)}"
        
        # Initialize job tracking
        dubbing_jobs[job_id] = {
            'status': 'queued',
            'progress': 0,
            'total_segments': len(segments),
            'completed_segments': 0,
            'current_chunk': 0,
            'video_path': request.video_path,
            'created_at': datetime.now().isoformat()
        }
        
        # Start background task
        background_tasks.add_task(
            process_dubbing_background,
            job_id,
            request.video_path,
            segments,
            request.voice
        )
        
        logger.info(f"🚀 Created dubbing job {job_id} for {request.video_path}")
        
        return {
            "status": "started",
            "job_id": job_id,
            "total_segments": len(segments),
            "message": "Dubbing started in background"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start dubbing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dub-progress/{job_id}")
async def dub_progress(job_id: str):
    """
    Server-Sent Events endpoint for real-time dubbing progress.
    Client connects and receives progress updates as they happen.
    """
    if job_id not in dubbing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        """Generate SSE events with job progress"""
        try:
            while True:
                job = dubbing_jobs.get(job_id)
                if not job:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Job not found"})
                    }
                    break
                
                # Send current progress
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "job_id": job_id,
                        "status": job['status'],
                        "progress": job['progress'],
                        "completed_segments": job['completed_segments'],
                        "total_segments": job['total_segments'],
                        "current_chunk": job.get('current_chunk', 0)
                    })
                }
                
                # If complete or error, send final message and close
                if job['status'] in ['complete', 'error']:
                    final_data = {
                        "job_id": job_id,
                        "status": job['status'],
                        "progress": 100 if job['status'] == 'complete' else job['progress']
                    }
                    
                    if job['status'] == 'complete':
                        final_data['audio_path'] = job.get('audio_path')
                        final_data['srt_path'] = job.get('srt_path')
                    elif job['status'] == 'error':
                        final_data['error'] = job.get('error', 'Unknown error')
                    
                    yield {
                        "event": "complete" if job['status'] == 'complete' else "error",
                        "data": json.dumps(final_data)
                    }
                    break
                
                # Wait before next update
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            logger.info(f"SSE connection closed for job {job_id}")
        except Exception as e:
            logger.error(f"SSE error for job {job_id}: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())


def smart_merge_segments(segments, min_duration=3.0, max_duration=10.0):
    """
    Smart merge segments for natural reading while keeping correct timeline.
    - Merges short segments into complete sentences
    - Keeps start time of first segment and end time of last segment in merge
    - Respects sentence boundaries (., !, ?)
    """
    if not segments:
        return segments
    
    merged = []
    current_group = []
    
    for seg in segments:
        current_group.append(seg)
        
        # Calculate group duration
        group_start = current_group[0]['start']
        group_end = seg['end']
        group_duration = group_end - group_start
        
        # Check if we should finalize this group
        text = seg['text'].strip()
        is_sentence_end = text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith('。')
        
        should_merge = (
            (is_sentence_end and group_duration >= min_duration) or
            group_duration >= max_duration
        )
        
        if should_merge and current_group:
            # Merge the group
            merged_text = ' '.join(s['text'].strip() for s in current_group)
            merged.append({
                'index': len(merged),
                'start': current_group[0]['start'],  # Keep original start
                'end': current_group[-1]['end'],     # Keep original end
                'text': merged_text,
                'original_indices': [s.get('index', i) for i, s in enumerate(current_group)]
            })
            current_group = []
    
    # Handle remaining segments
    if current_group:
        merged_text = ' '.join(s['text'].strip() for s in current_group)
        merged.append({
            'index': len(merged),
            'start': current_group[0]['start'],
            'end': current_group[-1]['end'],
            'text': merged_text,
            'original_indices': [s.get('index', i) for i, s in enumerate(current_group)]
        })
    
    return merged


# ====================
# INSTANT DUBBING ENDPOINT
# ====================

@app.post("/api/dub-instant")
async def dub_instant(request: DubbingRequest, background_tasks: BackgroundTasks):
    """
    Instant Dubbing - Generate first 5 segments immediately (~10 sec),
    then continue remaining in background.
    Returns audio URLs for first batch so playback can start immediately.
    """
    try:
        video_full_path = COURSES_BASE_DIR / request.video_path
        srt_path = video_full_path.with_suffix('.srt')
        
        if not srt_path.exists():
            raise HTTPException(status_code=404, detail="SRT file not found. Transcribe first.")
        
        # Parse SRT
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})\s+(.*?)(?=\s*\n\d+\s+\d{2}:\d{2}:\d{2}|$)', re.DOTALL)
        matches = pattern.findall(content)
        
        segments = []
        for m in matches:
            segments.append({
                'index': len(segments),
                'start': parse_srt_timestamp(m[1]),
                'end': parse_srt_timestamp(m[2]),
                'text': m[3].strip().replace('\n', ' ')
            })
        
        if not segments:
            raise HTTPException(status_code=400, detail="Could not parse SRT")
        
        # Create job directory in TEMP (auto-delete when server restarts)
        job_id = f"dub_{int(time.time() * 1000)}"
        video_full_path = COURSES_BASE_DIR / request.video_path
        video_dir = video_full_path.parent
        
        # Use temp directory for audio files - will be auto-deleted
        import tempfile
        job_dir = Path(tempfile.mkdtemp(prefix=f"dub_{job_id}_"))
        logger.info(f"📁 Using temp folder: {job_dir}")
        
        logger.info(f"⚡ Instant dubbing: {len(segments)} segments")

        # Translate ALL segments in batches (Gemini can handle ~30 at once)
        translated_segments = []
        if GEMINI_AVAILABLE:
            try:
                BATCH_SIZE = 30
                for batch_start in range(0, len(segments), BATCH_SIZE):
                    batch = segments[batch_start:batch_start + BATCH_SIZE]
                    prompt = f"Dịch các câu sau sang tiếng Việt tự nhiên cho lồng tiếng. Giữ nguyên định dạng JSON. Trả về duy nhất JSON array.\n\nNội dung: {json.dumps(batch, ensure_ascii=False)}"
                    response = gemini_model.generate_content(prompt)
                    match = re.search(r'\[.*\]', response.text, re.DOTALL)
                    if match:
                        batch_translated = json.loads(match.group())
                        translated_segments.extend(batch_translated)
                        logger.info(f"  Translated batch {batch_start//BATCH_SIZE + 1}: {len(batch_translated)} segments")
                    else:
                        # Fallback: use original for this batch
                        translated_segments.extend(batch)
            except Exception as ex:
                logger.error(f"Gemini error: {ex}")
        
        if not translated_segments or len(translated_segments) < len(segments):
            logger.warning(f"Translation incomplete: got {len(translated_segments)}/{len(segments)}, using original for missing")
            # Fill in missing segments
            while len(translated_segments) < len(segments):
                translated_segments.append(segments[len(translated_segments)])
        
        if not translated_segments:
            translated_segments = segments
        
        # SAVE VIETNAMESE SRT FILE (always save this)
        vi_srt_path = video_full_path.with_name(f"{video_full_path.stem}_vi.srt")
        try:
            with open(vi_srt_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(translated_segments, 1):
                    start_time = format_srt_time(seg['start'])
                    end_time = format_srt_time(seg['end'])
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{seg['text']}\n\n")
            logger.info(f"📝 Saved Vietnamese subtitles: {vi_srt_path}")
        except Exception as e:
            logger.error(f"Failed to save VI SRT: {e}")
            vi_srt_path = None
        
        # Generate FIRST 10 segments and CONCATENATE into one file
        FIRST_BATCH = min(10, len(translated_segments))
        engine = EdgeTTSEngine(voice=request.voice)
        audio_segments = []  # [(seg_info, audio_data)]
        
        for i in range(FIRST_BATCH):
            seg = translated_segments[i]
            try:
                audio_data = await engine.generate_audio(seg['text'], seg['start'], seg['end'])
                if audio_data and len(audio_data) > 0:
                    audio_segments.append((seg, audio_data))
                    logger.info(f"  ✓ Segment {i} generated ({seg['start']:.1f}s - {seg['end']:.1f}s)")
            except Exception as ex:
                logger.error(f"  ✗ Segment {i} failed: {ex}")
        
        if not audio_segments:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # CONCATENATE with proper timeline (silence padding)
        logger.info(f"🔧 Concatenating {len(audio_segments)} segments...")
        temp_files = []
        file_list = []
        current_position = 0.0
        
        try:
            for idx, (seg, audio_data) in enumerate(audio_segments):
                logger.info(f"  📝 Processing segment {idx} for saving...")
                
                # Add silence before segment if needed
                silence_duration = seg['start'] - current_position
                if silence_duration > 0.05:
                    silence_file = tempfile.NamedTemporaryFile(suffix=f"_sil_{idx}.mp3", delete=False)
                    silence_file.close()
                    temp_files.append(silence_file.name)
                    
                    silence_cmd = [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", f"{silence_duration:.6f}",
                        "-q:a", "9", "-acodec", "libmp3lame",
                        silence_file.name
                    ]
                    subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
                    file_list.append(silence_file.name)
                    current_position += silence_duration
                
                # Save segment audio to PERMANENT file (not temp)
                seg_file = job_dir / f"segment_{idx:03d}.mp3"
                logger.info(f"  💾 Writing {len(audio_data)} bytes to {seg_file}...")
                with open(seg_file, 'wb') as f:
                    f.write(audio_data)
                
                if seg_file.exists():
                    logger.info(f"  ✅ Saved segment {idx}: {seg_file.stat().st_size} bytes")
                else:
                    logger.error(f"  ❌ Failed to save segment {idx}")
                    
                file_list.append(str(seg_file))
                
                # Get segment duration
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(seg_file)],
                    capture_output=True, text=True, timeout=5
                )
                seg_duration = float(probe.stdout.strip()) if probe.stdout.strip() else (seg['end'] - seg['start'])
                current_position += seg_duration
            
            # Concatenate all
            concat_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            for f in file_list:
                concat_list_file.write(f"file '{f}'\n")
            concat_list_file.close()
            temp_files.append(concat_list_file.name)
            
            output_file = job_dir / "dubbed_partial.mp3"
            
            # Simple concat - the aresample filter was causing silent output!
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_file.name,
                "-acodec", "libmp3lame",
                "-b:a", "192k",
                str(output_file)
            ]
            result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)


            
            if result.returncode != 0:
                logger.error(f"FFmpeg concat error: {result.stderr.decode()}")
            else:
                logger.info(f"✅ Concatenated audio: {output_file} ({output_file.stat().st_size} bytes)")
            
        finally:
            for f in temp_files:
                try:
                    os.remove(f)
                except:
                    pass
        
        # Calculate end time of first batch
        last_seg = audio_segments[-1][0] if audio_segments else {'end': 30}
        
        # Calculate audio URL - use /.temp/ endpoint for temp directory
        audio_base_url = f"/.temp/{job_id}"
        
        # Initialize job tracking
        dubbing_jobs[job_id] = {
            'status': 'playing',
            'progress': int((FIRST_BATCH / len(translated_segments)) * 100),
            'total_segments': len(translated_segments),
            'completed_segments': FIRST_BATCH,
            'ready_segments': list(range(FIRST_BATCH)),
            'video_path': request.video_path,
            'job_dir': str(job_dir),
            'audio_base_url': audio_base_url,
            'translated_segments': translated_segments,
            'segments': segments,
            'voice': request.voice,
            'audio_file': str(job_dir / "dubbed_partial.mp3"),
            'audio_end_time': last_seg['end'],
            'created_at': datetime.now().isoformat()
        }
        
        # Start background task for remaining segments
        if FIRST_BATCH < len(translated_segments):
            background_tasks.add_task(
                process_remaining_segments,
                job_id,
                translated_segments[FIRST_BATCH:],
                FIRST_BATCH,
                request.voice,
                str(job_dir)
            )
        
        logger.info(f"⚡ Instant dubbing ready: {FIRST_BATCH}/{len(translated_segments)} segments, job {job_id}")
        
        return {
            "status": "ready",
            "job_id": job_id,
            "ready_segments": FIRST_BATCH,
            "total_segments": len(translated_segments),
            "audio_url": f"{audio_base_url}/dubbed_partial.mp3",
            "audio_end_time": last_seg['end'],
            "vi_srt_url": f"/media/{vi_srt_path.relative_to(COURSES_BASE_DIR)}".replace("\\", "/") if vi_srt_path else None,
            "message": f"First {FIRST_BATCH} segments ready! Vietnamese subtitles saved."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Instant dubbing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Request, Response

@app.api_route("/.temp/{job_id}/{filename}", methods=["GET", "HEAD"])
async def serve_temp_audio(job_id: str, filename: str, request: Request):
    """Serve temporary audio files with Range request support for streaming"""
    # Find job by matching partial ID (since job_id might have extra suffix)
    matching_job = None
    for jid in dubbing_jobs:
        if job_id in jid or jid in job_id:
            matching_job = dubbing_jobs[jid]
            break
    
    if not matching_job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job_dir = Path(matching_job['job_dir'])
    file_path = job_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    file_size = file_path.stat().st_size
    
    # Handle HEAD request
    if request.method == "HEAD":
        return Response(
            content=b"",
            status_code=200,
            headers={
                "Content-Type": "audio/mpeg",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes"
            }
        )
    
    # Handle Range requests for streaming
    range_header = request.headers.get("Range")
    
    if range_header:
        # Parse range header: bytes=0-1023
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1
            
            with open(file_path, 'rb') as f:
                f.seek(start)
                data = f.read(length)
            
            return Response(
                content=data,
                status_code=206,  # Partial Content
                headers={
                    "Content-Type": "audio/mpeg",
                    "Content-Length": str(length),
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes"
                }
            )
    
    # Full file response
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )


async def concat_available_segments(job_id: str, job_dir_path: Path):
    """Concatenate all available segments into a single audio file"""
    try:
        all_segments = dubbing_jobs[job_id].get('translated_segments', [])
        ready_indices = sorted(dubbing_jobs[job_id].get('ready_segments', []))
        
        if not ready_indices:
            return
        
        logger.info(f"🔧 Updating audio with {len(ready_indices)} segments...")
        
        temp_files = []
        file_list = []
        current_position = 0.0
        
        for idx in ready_indices:
            seg_file = job_dir_path / f"segment_{idx:03d}.mp3"
            if not seg_file.exists():
                continue
            
            seg = all_segments[idx] if idx < len(all_segments) else {'start': current_position, 'end': current_position + 2}
            
            # Add silence before segment if needed
            silence_duration = seg['start'] - current_position
            if silence_duration > 0.05:
                silence_file = tempfile.NamedTemporaryFile(suffix=f"_sil_{idx}.mp3", delete=False)
                silence_file.close()
                temp_files.append(silence_file.name)
                
                silence_cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", f"{silence_duration:.6f}",
                    "-q:a", "9", "-acodec", "libmp3lame",
                    silence_file.name
                ]
                subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
                file_list.append(silence_file.name)
                current_position += silence_duration
            
            file_list.append(str(seg_file))
            
            # Get segment duration
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(seg_file)],
                capture_output=True, text=True, timeout=5
            )
            seg_duration = float(probe.stdout.strip()) if probe.stdout.strip() else (seg['end'] - seg['start'])
            current_position += seg_duration
        
        if not file_list:
            return
        
        # Concatenate all
        concat_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for f in file_list:
            concat_list_file.write(f"file '{f}'\n")
        concat_list_file.close()
        temp_files.append(concat_list_file.name)
        
        output_file = job_dir_path / "dubbed_partial.mp3"
        
        # Simple concat - aresample filter was causing silent output!
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_file.name,
            "-acodec", "libmp3lame", 
            "-b:a", "192k",
            str(output_file)
        ]
        result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)


        
        if result.returncode == 0 and output_file.exists():
            size = output_file.stat().st_size
            dubbing_jobs[job_id]['audio_updated'] = True
            dubbing_jobs[job_id]['audio_version'] = dubbing_jobs[job_id].get('audio_version', 0) + 1
            logger.info(f"✅ Audio updated: {size} bytes, version {dubbing_jobs[job_id]['audio_version']}")
        else:
            logger.error(f"FFmpeg concat error: {result.stderr.decode()[:200]}")
        
        # Cleanup temp files
        for f in temp_files:
            try:
                os.remove(f)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Concat error: {e}")


async def process_single_segment(job_id: str, seg: dict, actual_index: int, engine: EdgeTTSEngine, job_dir_path: Path):
    """Helper to process a single segment with aggressive retry"""
    max_retries = 10
    for retry in range(max_retries):
        try:
            audio_data = await engine.generate_audio(seg['text'], seg['start'], seg['end'])
            if audio_data and len(audio_data) > 0:
                seg_file = job_dir_path / f"segment_{actual_index:03d}.mp3"
                with open(seg_file, 'wb') as f:
                    f.write(audio_data)
                
                # Update job status (thread-safe enough for dict updates in asyncio)
                dubbing_jobs[job_id]['completed_segments'] = max(dubbing_jobs[job_id]['completed_segments'], actual_index + 1)
                if actual_index not in dubbing_jobs[job_id]['ready_segments']:
                    dubbing_jobs[job_id]['ready_segments'].append(actual_index)
                
                # Update progress roughly
                total = dubbing_jobs[job_id]['total_segments']
                completed_count = len(dubbing_jobs[job_id]['ready_segments'])
                dubbing_jobs[job_id]['progress'] = int((completed_count / total) * 100)
                
                logger.info(f"  ✓ Segment {actual_index} ready (Parallel)")
                return True
            else:
                logger.warning(f"  ⚠ Segment {actual_index} empty audio (attempt {retry + 1})")
                if retry < max_retries - 1:
                    await asyncio.sleep(1.0 * (retry + 1))
        except Exception as ex:
            logger.error(f"  ✗ Segment {actual_index} failed (attempt {retry + 1}): {ex}")
            if retry < max_retries - 1:
                await asyncio.sleep(1.0 * (retry + 1))
    return False

async def process_remaining_segments(job_id: str, remaining_segments: list, start_index: int, voice: str, job_dir: str):
    """Background task to generate remaining segments with parallel processing"""
    try:
        logger.info(f"🔄 Processing remaining {len(remaining_segments)} segments for {job_id} (Parallel Mode)...")
        engine = EdgeTTSEngine(voice=voice)
        job_dir_path = Path(job_dir)
        
        BATCH_SIZE = 3  # Process 3 segments at a time for speed
        
        for i in range(0, len(remaining_segments), BATCH_SIZE):
            batch = remaining_segments[i : i + BATCH_SIZE]
            tasks = []
            
            # Create tasks for this batch
            for j, seg in enumerate(batch):
                actual_index = start_index + i + j
                tasks.append(process_single_segment(job_id, seg, actual_index, engine, job_dir_path))
            
            # Run batch in parallel
            await asyncio.gather(*tasks)
            
            # Update consolidated audio file after every batch
            await concat_available_segments(job_id, job_dir_path)




        
        # Now concatenate ALL segments into full audio file
        logger.info(f"🔧 Concatenating all segments into full audio...")
        all_segments = dubbing_jobs[job_id].get('translated_segments', [])
        temp_files = []
        file_list = []
        current_position = 0.0
        
        try:
            for idx, seg in enumerate(all_segments):
                seg_file = job_dir_path / f"segment_{idx:03d}.mp3"
                if not seg_file.exists():
                    logger.warning(f"  Segment {idx} file not found, skipping")
                    continue
                
                # Add silence before segment if needed
                silence_duration = seg['start'] - current_position
                if silence_duration > 0.05:
                    silence_file = tempfile.NamedTemporaryFile(suffix=f"_sil_{idx}.mp3", delete=False)
                    silence_file.close()
                    temp_files.append(silence_file.name)
                    
                    silence_cmd = [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", f"{silence_duration:.6f}",
                        "-q:a", "9", "-acodec", "libmp3lame",
                        silence_file.name
                    ]
                    subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
                    file_list.append(silence_file.name)
                    current_position += silence_duration
                
                file_list.append(str(seg_file))
                
                # Get segment duration
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(seg_file)],
                    capture_output=True, text=True, timeout=5
                )
                seg_duration = float(probe.stdout.strip()) if probe.stdout.strip() else (seg['end'] - seg['start'])
                current_position += seg_duration
            
            # Concatenate all
            concat_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            for f in file_list:
                concat_list_file.write(f"file '{f}'\n")
            concat_list_file.close()
            temp_files.append(concat_list_file.name)
            
            output_file = job_dir_path / "dubbed_full.mp3"
            
            # Use consistent audio format for smooth playback
            output_file = job_dir_path / "dubbed_full.mp3"
            
            # Simple concat - aresample filter was causing silent output!
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_file.name,
                "-acodec", "libmp3lame",
                "-b:a", "192k",
                str(output_file)
            ]
            result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)


            
            if result.returncode == 0 and output_file.exists():
                logger.info(f"✅ Full audio created: {output_file} ({output_file.stat().st_size} bytes)")
                dubbing_jobs[job_id]['full_audio_url'] = f"/.temp/{job_id}/dubbed_full.mp3"
                
                # SAVE TO PERMANENT FILE
                video_path = dubbing_jobs[job_id].get('video_path', '')
                if video_path:
                    video_full_path = COURSES_BASE_DIR / video_path
                    permanent_audio = video_full_path.with_name(f"{video_full_path.stem}_dubbed.mp3")
                    try:
                        import shutil
                        shutil.copy(output_file, permanent_audio)
                        logger.info(f"💾 Saved permanent audio: {permanent_audio}")
                    except Exception as e:
                        logger.error(f"Failed to save permanent audio: {e}")
                    
                    # GENERATE VIETNAMESE SUBTITLES
                    vi_srt_path = video_full_path.with_name(f"{video_full_path.stem}_vi.srt")
                    try:
                        with open(vi_srt_path, 'w', encoding='utf-8') as f:
                            for i, seg in enumerate(all_segments, 1):
                                start_time = format_srt_time(seg['start'])
                                end_time = format_srt_time(seg['end'])
                                f.write(f"{i}\n")
                                f.write(f"{start_time} --> {end_time}\n")
                                f.write(f"{seg['text']}\n\n")
                        logger.info(f"📝 Created Vietnamese subtitles: {vi_srt_path}")
                    except Exception as e:
                        logger.error(f"Failed to create VI subtitles: {e}")
            else:
                logger.error(f"FFmpeg concat error: {result.stderr.decode()}")
            
        finally:
            for f in temp_files:
                try:
                    os.remove(f)
                except:
                    pass
        
        dubbing_jobs[job_id]['status'] = 'complete'
        logger.info(f"✅ All segments complete for {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Background processing error: {e}")
        dubbing_jobs[job_id]['status'] = 'error'
        dubbing_jobs[job_id]['error'] = str(e)


@app.get("/api/dub-instant-progress/{job_id}")
async def dub_instant_progress(job_id: str):
    """SSE endpoint for instant dubbing progress - notifies when new segments are ready"""
    if job_id not in dubbing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        last_ready_count = 0
        try:
            while True:
                job = dubbing_jobs.get(job_id)
                if not job:
                    break
                
                ready_segments = job.get('ready_segments', [])
                
                # Notify about newly ready segments
                if len(ready_segments) > last_ready_count:
                    translated = job.get('translated_segments', [])
                    for idx in ready_segments[last_ready_count:]:
                        seg_timing = translated[idx] if idx < len(translated) else {}
                        yield {
                            "event": "segment_ready",
                            "data": json.dumps({
                                "segment_index": idx,
                                "url": f"/.temp/{job_id}/segment_{idx:03d}.mp3",
                                "start": seg_timing.get('start', 0),
                                "end": seg_timing.get('end', 0),
                                "progress": job['progress']
                            })
                        }
                    last_ready_count = len(ready_segments)
                
                # Notify when audio file has been updated
                if job.get('audio_updated'):
                    dubbing_jobs[job_id]['audio_updated'] = False  # Reset flag
                    audio_base = job.get('audio_base_url', f"/.temp/{job_id}")
                    yield {
                        "event": "audio_updated",
                        "data": json.dumps({
                            "audio_url": f"{audio_base}/dubbed_partial.mp3",
                            "version": job.get('audio_version', 1),
                            "ready_count": len(ready_segments),
                            "progress": job['progress']
                        })
                    }
                
                if job['status'] == 'complete':
                    audio_base = job.get('audio_base_url', f"/.temp/{job_id}")
                    yield {
                        "event": "complete",
                        "data": json.dumps({
                            "job_id": job_id, 
                            "total_segments": job['total_segments'],
                            "full_audio_url": job.get('full_audio_url', f"{audio_base}/dubbed_full.mp3")
                        })
                    }
                    break
                elif job['status'] == 'error':
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": job.get('error', 'Unknown error')})
                    }
                    break
                
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
    
    return EventSourceResponse(event_generator())


# Serve temp files
@app.get("/.temp/{job_id}/{filename}")
async def serve_temp_audio(job_id: str, filename: str):
    """Serve temporary audio segment files"""
    file_path = COURSES_BASE_DIR / ".temp" / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Segment not found")
    return FileResponse(file_path, media_type="audio/mpeg")


async def process_dubbing_background(job_id: str, video_path: str, segments: list, voice: str):
    """
    Background worker that processes dubbing in chunks.
    Updates dubbing_jobs[job_id] as progress is made.
    """
    try:
        logger.info(f"🚀 Processing job {job_id}")
        dubbing_jobs[job_id]['status'] = 'processing'
        
        video_full_path = COURSES_BASE_DIR / video_path
        CHUNK_SIZE = 10
        
        # Translate
        logger.info(f"📝 Translating {len(segments)} segments...")
        translated_segments = []
        
        if GEMINI_AVAILABLE:
            merged = smart_merge_subtitles(segments)
            chunk_size = 20
            for i in range(0, len(merged), chunk_size):
                chunk = merged[i:i+chunk_size]
                prompt = f"Dịch các câu sau sang tiếng Việt tự nhiên cho lồng tiếng. Giữ nguyên định dạng JSON array [{{'start': float, 'end': float, 'text': string}}]. Trả về duy nhất JSON array.\n\nNội dung: {json.dumps(chunk, ensure_ascii=False)}"
                
                try:
                    response = gemini_model.generate_content(prompt)
                    match = re.search(r'\[.*\]', response.text, re.DOTALL)
                    if match:
                        translated_chunk = json.loads(match.group())
                        translated_segments.extend(translated_chunk)
                    else:
                        translated_segments.extend(chunk)
                except Exception as ex:
                    logger.error(f"Gemini error: {ex}")
                    translated_segments.extend(chunk)
        else:
            translated_segments = segments
        
        # Generate audio in chunks
        logger.info(f"🎵 Generating audio...")
        engine = EdgeTTSEngine(voice=voice)
        all_audio_segments = []
        
        for chunk_idx in range(0, len(translated_segments), CHUNK_SIZE):
            chunk_end = min(chunk_idx + CHUNK_SIZE, len(translated_segments))
            chunk = translated_segments[chunk_idx:chunk_end]
            
            # Generate audio for chunk
            tasks = [engine.generate_audio(s['text'], s['start'], s['end']) for s in chunk]
            audio_chunk = await asyncio.gather(*tasks)
            
            # Save segments
            for seg, audio in zip(chunk, audio_chunk):
                if audio and len(audio) > 0:
                    all_audio_segments.append((seg, audio))
            
            # Update progress
            dubbing_jobs[job_id]['completed_segments'] = chunk_end
            dubbing_jobs[job_id]['progress'] = int((chunk_end / len(translated_segments)) * 100)
            dubbing_jobs[job_id]['current_chunk'] = chunk_idx // CHUNK_SIZE
        
        # Concatenate audio (same logic as sync version)
        logger.info(f"🔧 Concatenating audio...")
        temp_files = []
        file_list = []
        
        try:
            current_position = 0.0
            
            for idx, (seg, audio_data) in enumerate(all_audio_segments):
                silence_duration = seg['start'] - current_position
                
                if silence_duration > 0.05:
                    silence_file = tempfile.NamedTemporaryFile(suffix=f"_silence_{idx}.mp3", delete=False)
                    silence_file.close()
                    temp_files.append(silence_file.name)
                    
                    silence_cmd = [
                        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", f"{silence_duration:.6f}",
                        "-q:a", "9", "-acodec", "libmp3lame",
                        silence_file.name
                    ]
                    result = subprocess.run(silence_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
                    if result.returncode == 0:
                        file_list.append(silence_file.name)
                        current_position += silence_duration
                
                audio_file = tempfile.NamedTemporaryFile(suffix=f"_audio_{idx}.mp3", delete=False)
                audio_file.write(audio_data)
                audio_file.close()
                temp_files.append(audio_file.name)
                file_list.append(audio_file.name)
                current_position = seg['end']
            
            concat_file = tempfile.NamedTemporaryFile(mode='w', suffix='_concat.txt', delete=False, encoding='utf-8')
            for f in file_list:
                escaped_path = f.replace("'", "'\\''")
                concat_file.write(f"file '{escaped_path}'\n")
            concat_file.close()
            temp_files.append(concat_file.name)
            
            dubbed_audio_path = video_full_path.with_name(f"{video_full_path.stem}_dubbed.mp3")
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file.name,
                "-acodec", "libmp3lame", "-q:a", "2",
                str(dubbed_audio_path)
            ]
            
            result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr.decode()}")
            
        finally:
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
        
        # Generate SRT
        vi_srt_path = video_full_path.with_name(f"{video_full_path.stem}_vi.srt")
        srt_content = ""
        for idx, seg in enumerate(translated_segments, 1):
            start_time = format_srt_time(seg['start'])
            end_time = format_srt_time(seg['end'])
            srt_content += f"{idx}\n{start_time} --> {end_time}\n{seg['text']}\n\n"
        
        with open(vi_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        # Mark complete
        rel_audio_path = str(dubbed_audio_path.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
        rel_srt_path = str(vi_srt_path.relative_to(COURSES_BASE_DIR)).replace('\\', '/')
        
        dubbing_jobs[job_id]['status'] = 'complete'
        dubbing_jobs[job_id]['progress'] = 100
        dubbing_jobs[job_id]['audio_path'] = rel_audio_path
        dubbing_jobs[job_id]['srt_path'] = rel_srt_path
        
        logger.info(f"✅ Job {job_id} complete!")
        
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        dubbing_jobs[job_id]['status'] = 'error'
        dubbing_jobs[job_id]['error'] = str(e)

def parse_srt_timestamp(timestamp: str) -> float:
    """Robust SRT timestamp parser (handles , or .)"""
    timestamp = timestamp.strip().replace(',', '.')
    parts = timestamp.split(':')
    if len(parts) == 3:
        # Expected format HH:MM:SS.mmm
        h, m, s = parts
        return int(h)*3600 + int(m)*60 + float(s)
    return 0.0

def smart_merge_subtitles(subtitles: list) -> list:
    if not subtitles: return []
    result = []
    i = 0
    while i < len(subtitles):
        curr = subtitles[i].copy()
        text = curr['text']
        start = curr['start']
        end = curr['end']
        j = i + 1
        # Merge if short or doesn't end with punctuation
        while j < len(subtitles) and (len(text.split()) < 8) and not any(text.endswith(p) for p in '.!?'):
            next_seg = subtitles[j]
            text += " " + next_seg['text']
            end = next_seg['end']
            j += 1
            if len(text.split()) > 20: break
        result.append({'start': start, 'end': end, 'text': text})
        i = j
    return result


# ====== FOLDER MANAGEMENT ENDPOINTS ======

import platform

class FolderScanRequest(BaseModel):
    folder_path: str

class OpenFolderRequest(BaseModel):
    folder_path: str

@app.post("/api/select-folder")
async def select_folder():
    """Open folder dialog for user to select media folder"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        folder_path = filedialog.askdirectory(
            title="Chọn folder chứa videos, SRT và bài tập",
            initialdir=str(Path.home())
        )
        root.destroy()
        
        if folder_path:
            return JSONResponse({
                "status": "success",
                "folder_path": folder_path
            })
        return JSONResponse({"status": "cancelled"})
        
    except Exception as e:
        print(f"❌ Folder selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan-folder")
async def scan_folder(request: FolderScanRequest):
    """Scan folder for videos and create course structure"""
    folder_path = Path(request.folder_path)
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    try:
        videos = []
        exercises = []
        video_extensions = {'.mp4', '.mkv', '.avi', '.webm', '.mov'}
        exercise_extensions = {'.pdf', '.docx', '.txt', '.ipynb', '.html'}
        
        # Scan folder recursively
        for root, dirs, files in os.walk(folder_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()
                file_name = file_path.stem
                
                # Extract number from filename
                num_match = re.search(r'^(\d+)', file_name)
                order = int(num_match.group(1)) if num_match else 999
                
                if file_ext in video_extensions:
                    # Check for matching SRT
                    srt_path = file_path.with_suffix('.srt')
                    has_srt = srt_path.exists()
                    
                    # Clean title
                    title = re.sub(r'^\d+\s*[-_]?\s*', '', file_name)
                    title = title.replace('_', ' ').replace('-', ' ').strip()
                    
                    videos.append({
                        "order": order,
                        "title": title or file_name,
                        "path": str(file_path.relative_to(folder_path)),
                        "subtitle": str(srt_path.relative_to(folder_path)) if has_srt else None,
                        "type": "video",
                        "has_srt": has_srt,
                        "filename": file
                    })
                
                elif file_ext in exercise_extensions:
                    title = re.sub(r'^\d+\s*[-_]?\s*', '', file_name)
                    title = title.replace('_', ' ').replace('-', ' ').strip()
                    
                    exercises.append({
                        "order": order,
                        "title": title or file_name,
                        "path": str(file_path.relative_to(folder_path)),
                        "type": "exercise",
                        "filename": file
                    })
        
        # Sort by order
        videos.sort(key=lambda x: x['order'])
        exercises.sort(key=lambda x: x['order'])
        
        # Update COURSES_BASE_DIR to selected folder
        global COURSES_BASE_DIR
        COURSES_BASE_DIR = folder_path
        
        # Save folder path to config file
        config_file = PLATFORM_DIR / "folder_config.txt"
        with open(config_file, 'w') as f:
            f.write(str(folder_path))
        
        print(f"📁 Scanned folder: {folder_path}")
        print(f"   Videos: {len(videos)} ({sum(1 for v in videos if v['has_srt'])} with SRT)")
        print(f"   Exercises: {len(exercises)}")
        
        return JSONResponse({
            "status": "success",
            "video_count": len(videos),
            "exercise_count": len(exercises),
            "videos_with_srt": sum(1 for v in videos if v['has_srt']),
            "lessons": videos + exercises
        })
        
    except Exception as e:
        print(f"❌ Folder scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/open-folder")
async def open_folder(request: OpenFolderRequest):
    """Open folder in system file explorer"""
    folder_path = Path(request.folder_path)
    
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(str(folder_path))
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', str(folder_path)])
        else:  # Linux
            subprocess.run(['xdg-open', str(folder_path)])
        
        return JSONResponse({"status": "success"})
    except Exception as e:
        print(f"❌ Open folder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====== PROGRESS TRACKING ======

PROGRESS_FILE = PLATFORM_DIR / "progress.json"

def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_progress(progress_data):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

class ProgressUpdateRequest(BaseModel):
    lesson_path: str
    completed: bool

@app.get("/api/progress")
async def get_all_progress():
    """Lấy toàn bộ dữ liệu tiến độ học tập"""
    return load_progress()

@app.post("/api/update-progress")
async def update_progress(request: ProgressUpdateRequest):
    """Cập nhật trạng thái hoàn thành của một bài học"""
    progress = load_progress()
    progress[request.lesson_path] = {
        "completed": request.completed,
        "updated_at": datetime.now().isoformat()
    }
    save_progress(progress)
    return {"status": "success", "lesson": request.lesson_path, "completed": request.completed}

@app.get("/api/stats")
async def get_stats():
    """Tính toán phần trăm hoàn thành và thống kê chung"""
    # Scan lại courses để có danh sách bài học hiện tại
    courses_result = await list_courses()
    courses = courses_result['courses']
    
    progress = load_progress()
    
    stats = {
        "total_videos": 0,
        "completed_videos": 0,
        "total_exercises": 0,
        "completed_exercises": 0,
        "overall_percentage": 0,
        "courses_stats": []
    }
    
    for course in courses:
        lessons_result = await list_lessons(course['lessons_path'])
        lessons = lessons_result['lessons']
        
        course_video_count = 0
        course_completed_count = 0
        
        for lesson in lessons:
            if lesson['type'] == 'video':
                stats["total_videos"] += 1
                course_video_count += 1
                if progress.get(lesson['path'], {}).get('completed'):
                    stats["completed_videos"] += 1
                    course_completed_count += 1
            elif lesson['type'] in ['exercise', 'html', 'pdf']:
                stats["total_exercises"] += 1
                if progress.get(lesson['path'], {}).get('completed'):
                    stats["completed_exercises"] += 1
        
        percentage = (course_completed_count / course_video_count * 100) if course_video_count > 0 else 0
        
        stats["courses_stats"].append({
            "id": course['id'],
            "name": course['name'],
            "video_count": course_video_count,
            "completed_count": course_completed_count,
            "percentage": round(percentage, 1)
        })
        
    total_total = stats["total_videos"] + stats["total_exercises"]
    total_completed = stats["completed_videos"] + stats["completed_exercises"]
    stats["overall_percentage"] = round((total_completed / total_total * 100), 1) if total_total > 0 else 0
    
    return {**stats, "progress": progress}

# ====== END PROGRESS TRACKING ======

# Serve static files (REMOVED as frontend was deleted)
# app.mount("/css", StaticFiles(directory=str(PLATFORM_DIR / "css")), name="css")
# app.mount("/js", StaticFiles(directory=str(PLATFORM_DIR / "js")), name="js")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve API status"""
    return HTMLResponse(content="<h1>AI Learning Platform API</h1><p>Backend is running. Use /api/courses to list content.</p>")

if __name__ == "__main__":
    import uvicorn
    print("🎓 AI Learning Platform Server")
    print(f"📁 Courses directory: {COURSES_BASE_DIR}")
    print(f"🤖 Gemini AI: {'✅ Available' if GEMINI_AVAILABLE else '❌ Not configured'}")
    print("🌐 Open http://localhost:3000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=3000)


