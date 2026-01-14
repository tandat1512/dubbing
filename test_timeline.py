import requests
import json
import os
from mutagen.mp3 import MP3

API_BASE = "http://localhost:3000/api"

def test_timeline_sync():
    video_path = "04 Pandas Basics (DataFrame Basics I)-20251224T123346Z-1-001/04 Pandas Basics (DataFrame Basics I)/001 Create your very first Pandas DataFrame (from csv).mp4"
    print(f"🎬 Testing Timeline-Accurate Dubbing for: {video_path}")
    
    try:
        # Delete old dubbed file if exists
        old_dubbed = f"x:/youtube/{video_path.replace('.mp4', '_dubbed.mp3')}"
        if os.path.exists(old_dubbed):
            os.remove(old_dubbed)
            print(f"🗑️ Deleted old dubbed file")
        
        # Request new dubbing
        print("⏳ Creating new dubbed audio with silence padding...")
        res = requests.post(f"{API_BASE}/dub", json={"video_path": video_path}, timeout=300)
        print(f"✅ Status Code: {res.status_code}")
        data = res.json()
        print(f"📊 Response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Check dubbed audio duration
        if 'audio_path' in data:
            audio_full_path = f"x:/youtube/{data['audio_path']}"
            if os.path.exists(audio_full_path):
                audio = MP3(audio_full_path)
                duration = audio.info.length
                print(f"🎵 Dubbed audio duration: {duration:.2f}s")
                
                # Compare with SRT file
                srt_path = f"x:/youtube/{video_path.replace('.mp4', '.srt')}"
                if os.path.exists(srt_path):
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Find last timestamp
                        for line in reversed(lines):
                            if '-->' in line:
                                end_time = line.split('-->')[1].strip().split(',')[0]
                                h, m, s = end_time.split(':')
                                srt_duration = int(h)*3600 + int(m)*60 + float(s)
                                print(f"📝 Original SRT duration: {srt_duration:.2f}s")
                                diff = duration - srt_duration
                                print(f"⏱️ Timeline difference: {diff:.2f}s")
                                if abs(diff) < 1.0:
                                    print("✅ Timeline sync is EXCELLENT (< 1s difference)")
                                elif abs(diff) < 3.0:
                                    print("⚠️ Timeline sync is acceptable (< 3s difference)")
                                else:
                                    print("❌ Timeline sync needs improvement (> 3s difference)")
                                break
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_timeline_sync()
