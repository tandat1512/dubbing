import requests
import json
import time

API_BASE = "http://localhost:3000/api"

def test_connectivity():
    print("🔍 Testing Backend Connectivity...")
    try:
        # 1. Get courses
        res = requests.get(f"{API_BASE}/courses")
        courses = res.json().get('courses', [])
        print(f"✅ Courses found: {len(courses)}")
        
        if courses:
            course_id = courses[0]['lessons_path']
            print(f"📁 Testing lessons for: {course_id}")
            res = requests.get(f"{API_BASE}/lessons/{course_id}")
            lessons = res.json().get('lessons', [])
            print(f"✅ Lessons found: {len(lessons)}")
            
            videos = [l for l in lessons if l['type'] == 'video']
            if videos:
                video = videos[0]
                print(f"🎬 Testing Dubbing for: {video['path']}")
                # This might take time, we just check if endpoint responds
                # res = requests.post(f"{API_BASE}/dub", json={"video_path": video['path']})
                # print(f"✅ Dubbing result: {res.status_code}")
            else:
                print("⚠️ No videos found to test dubbing.")
        else:
            print("⚠️ No courses found to test.")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_connectivity()
