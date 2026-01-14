import requests
import json
import os
import subprocess

API_BASE = "http://localhost:3000/api"

def get_audio_duration(file_path):
    """Get audio duration using ffprobe"""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except:
        return None
    return None

def test_timeline_sync():
    video_path = "04 Pandas Basics (DataFrame Basics I)-20251224T123346Z-1-001/04 Pandas Basics (DataFrame Basics I)/001 Create your very first Pandas DataFrame (from csv).mp4"
    print(f"🎬 Testing Timeline-Accurate Dubbing")
    print(f"📹 Video: {video_path[:50]}...")
    
    try:
        # Delete old dubbed file if exists
        old_dubbed = f"x:/youtube/{video_path.replace('.mp4', '_dubbed.mp3')}"
        if os.path.exists(old_dubbed):
            os.remove(old_dubbed)
            print(f"🗑️ Deleted old dubbed file")
        
        # Request new dubbing
        print("⏳ Creating dubbed audio with silence padding...")
        res = requests.post(f"{API_BASE}/dub", json={"video_path": video_path}, timeout=300)
        print(f"✅ Response status: {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            print(f"📊 Segments processed: {data.get('segments', 0)}")
            
            # Check dubbed audio duration
            if 'audio_path' in data:
                audio_full_path = f"x:/youtube/{data['audio_path']}"
                if os.path.exists(audio_full_path):
                    duration = get_audio_duration(audio_full_path)
                    if duration:
                        print(f"🎵 Dubbed audio duration: {duration:.2f}s (~{duration/60:.1f} minutes)")
                        
                        # Compare with SRT file
                        srt_path = f"x:/youtube/{video_path.replace('.mp4', '.srt')}"
                        if os.path.exists(srt_path):
                            with open(srt_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                # Find last timestamp
                                for line in reversed(lines):
                                    if '-->' in line:
                                        end_time = line.split('-->')[1].strip()
                                        # Parse 00:06:57,810
                                        time_part = end_time.split(',')[0]
                                        h, m, s = time_part.split(':')
                                        ms = end_time.split(',')[1] if ',' in end_time else '0'
                                        srt_duration = int(h)*3600 + int(m)*60 + float(s) + float(ms)/1000
                                        print(f"📝 Original SRT duration: {srt_duration:.2f}s (~{srt_duration/60:.1f} minutes)")
                                        diff = duration - srt_duration
                                        print(f"⏱️ Timeline difference: {diff:+.2f}s")
                                        
                                        if abs(diff) < 1.0:
                                            print("✅ ✅ ✅ Timeline sync is EXCELLENT (< 1s difference)")
                                        elif abs(diff) < 3.0:
                                            print("⚠️ Timeline sync is acceptable (< 3s difference)")
                                        else:
                                            print("❌ Timeline sync needs improvement (> 3s difference)")
                                        break
                    else:
                        print("❌ Could not measure audio duration (ffprobe failed)")
                else:
                    print(f"❌ Dubbed audio file not found: {audio_full_path}")
        else:
            print(f"❌ Dubbing request failed: {res.text}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_timeline_sync()
