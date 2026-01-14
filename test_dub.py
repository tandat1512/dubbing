import requests
import json

API_BASE = "http://localhost:3000/api"

def test_dub():
    video_path = "04 Pandas Basics (DataFrame Basics I)-20251224T123346Z-1-001/04 Pandas Basics (DataFrame Basics I)/001 Create your very first Pandas DataFrame (from csv).mp4"
    print(f"🎬 Testing Dubbing for: {video_path}")
    
    try:
        res = requests.post(f"{API_BASE}/dub", json={"video_path": video_path})
        print(f"Status Code: {res.status_code}")
        data = res.json()
        print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"❌ Dubbing test failed: {e}")

if __name__ == "__main__":
    test_dub()
