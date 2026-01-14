# Backend API Requirements for Folder Management

## 1. Select Folder API
```python
@app.route('/api/select-folder', methods=['POST'])
def select_folder():
    """Open folder dialog for user to select media folder"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Keep window on top
    
    folder_path = filedialog.askdirectory(
        title="Chọn folder chứa videos, SRT và bài tập",
        initialdir=os.path.expanduser("~")
    )
    root.destroy()
    
    if folder_path:
        return jsonify({
            "status": "success",
            "folder_path": folder_path
        })
    return jsonify({"status": "cancelled"})
```

## 2. Scan Folder API
```python
@app.route('/api/scan-folder', methods=['POST'])
def scan_folder():
    """
    Scan folder for videos and create course structure
    - Find all video files (.mp4, .mkv, .avi, .webm)
    - Match with SRT files (same name)
    - Find exercise files (.pdf, .docx, .txt, .ipynb)
    - Create lessons with numbering
    """
    data = request.json
    folder_path = data.get('folder_path')
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({"status": "error", "message": "Invalid folder path"})
    
    videos = []
    exercises = []
    video_extensions = {'.mp4', '.mkv', '.avi', '.webm', '.mov'}
    exercise_extensions = {'.pdf', '.docx', '.txt', '.ipynb', '.html'}
    
    # Scan folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            file_name = os.path.splitext(file)[0]
            
            if file_ext in video_extensions:
                # Check for matching SRT
                srt_path = os.path.join(root, file_name + '.srt')
                has_srt = os.path.exists(srt_path)
                
                # Extract number from filename (e.g., "01-intro.mp4" -> 1)
                import re
                num_match = re.search(r'^(\d+)', file_name)
                order = int(num_match.group(1)) if num_match else len(videos)
                
                videos.append({
                    "order": order,
                    "title": file_name.replace('_', ' ').replace('-', ' ').title(),
                    "path": os.path.relpath(file_path, folder_path),
                    "subtitle": os.path.relpath(srt_path, folder_path) if has_srt else None,
                    "type": "video",
                    "has_srt": has_srt
                })
            
            elif file_ext in exercise_extensions:
                num_match = re.search(r'^(\d+)', file_name)
                order = int(num_match.group(1)) if num_match else len(exercises)
                
                exercises.append({
                    "order": order,
                    "title": file_name.replace('_', ' ').replace('-', ' ').title(),
                    "path": os.path.relpath(file_path, folder_path),
                    "type": "exercise"
                })
    
    # Sort by order
    videos.sort(key=lambda x: x['order'])
    exercises.sort(key=lambda x: x['order'])
    
    # Combine and save to course structure
    all_lessons = videos + exercises
    course_name = os.path.basename(folder_path)
    
    # Save to courses.json or database
    course_data = {
        "name": course_name,
        "folder_path": folder_path,
        "lessons": all_lessons,
        "video_count": len(videos),
        "exercise_count": len(exercises)
    }
    
    # Save to config
    save_course_data(course_data)
    
    return jsonify({
        "status": "success",
        "video_count": len(videos),
        "exercise_count": len(exercises),
        "lessons": all_lessons
    })
```

## 3. Open Folder API
```python
@app.route('/api/open-folder', methods=['POST'])
def open_folder():
    """Open folder in system file explorer"""
    import platform
    import subprocess
    
    data = request.json
    folder_path = data.get('folder_path')
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({"status": "error", "message": "Folder not found"})
    
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(folder_path)
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', folder_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_path])
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
```

## Usage Flow

1. **User clicks "Chọn Folder Media"**
   - Opens folder dialog
   - User selects folder
   - Backend scans folder for videos/exercises
   - Frontend reloads course list
   - Shows: "✅ Đã quét folder: X videos, Y bài tập"

2. **User clicks "Mở Folder"**
   - Opens currently selected folder in file explorer
   - User can view/edit files directly

3. **Sidebar displays**
   - Videos with SRT (có icon subtitle)
   - Videos without SRT (có warning icon)
   - Exercises (có icon assignment)
   - All sorted by number prefix
