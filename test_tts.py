import asyncio
import edge_tts

async def test_tts():
    print("Testing Edge TTS...")
    text = "Xin chào, đây là bài kiểm tra tiếng Việt"
    voice = "vi-VN-HoaiMyNeural"
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save("x:/youtube/tts_test.mp3")
        print("✅ Saved to x:/youtube/tts_test.mp3")
        
        # Check file size
        import os
        size = os.path.getsize("x:/youtube/tts_test.mp3")
        print(f"📁 File size: {size} bytes")
        
        if size > 1000:
            print("✅ File has content!")
        else:
            print("❌ File too small, may be empty")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tts())
