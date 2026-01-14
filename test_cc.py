import requests
import json
import os

API_BASE = "http://localhost:3000/api"

def test_dub_with_cc():
    video_path = "04 Pandas Basics (DataFrame Basics I)-20251224T123346Z-1-001/04 Pandas Basics (DataFrame Basics I)/001 Create your very first Pandas DataFrame (from csv).mp4"
    print(f"🎬 Testing Dubbing with Vietnamese CC for: {video_path}")
    
    try:
        res = requests.post(f"{API_BASE}/dub", json={"video_path": video_path})
        print(f"✅ Status Code: {res.status_code}")
        data = res.json()
        print(f"📊 Response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # Check if Vietnamese SRT was created
        if 'srt_path' in data:
            srt_full_path = f"x:/youtube/{data['srt_path']}"
            if os.path.exists(srt_full_path):
                print(f"✅ Vietnamese SRT created: {srt_full_path}")
                with open(srt_full_path, 'r', encoding='utf-8') as f:
                    preview = f.read(500)
                    print(f"📝 Preview:\n{preview}...")
            else:
                print(f"❌ Vietnamese SRT not found at: {srt_full_path}")
        
    except Exception as e:
        print(f"❌ Dubbing test failed: {e}")

if __name__ == "__main__":
    test_dub_with_cc()
