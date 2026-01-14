import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Library,
  Mic,
  Languages,
  CheckCircle2,
  FolderPlus,
  TrendingUp,
  PlayCircle,
  FileText,
  AlertCircle,
  Search,
  Settings,
  MoreVertical,
  RotateCw
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:3000/api';

function App() {
  const [activeTab, setActiveTab] = useState('courses');
  const [courses, setCourses] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [lessons, setLessons] = useState([]);
  const [videosWithoutSrt, setVideosWithoutSrt] = useState([]);
  const [currentLesson, setCurrentLesson] = useState(null);
  const [processingDub, setProcessingDub] = useState(false);
  const [processingSTT, setProcessingSTT] = useState(false);
  const [useDubbedAudio, setUseDubbedAudio] = useState(false);
  const [subtitles, setSubtitles] = useState([]);
  const [vietnameseSubtitles, setVietnameseSubtitles] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [showSubtitles, setShowSubtitles] = useState(true);
  const [subtitleLanguage, setSubtitleLanguage] = useState('en'); // 'en' or 'vi'
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Instant Dubbing state
  const [segmentAudios, setSegmentAudios] = useState({}); // {index: Audio element}
  const [dubbingProgressMap, setDubbingProgressMap] = useState({}); // { [path]: percentage }
  const [currentJobId, setCurrentJobId] = useState(null);

  // Folder Creation State
  const [newFolderName, setNewFolderName] = useState('');
  const [isCreatingFolder, setIsCreatingFolder] = useState(false);

  // Smart Sync Refs
  const pendingAudioUrlRef = useRef(null);
  const restorePositionRef = useRef(null); // For seamless audio reload
  const [segmentInfo, setSegmentInfo] = useState([]); // [{index, start, end, url}]
  const [dubbedAudioUrl, setDubbedAudioUrl] = useState(''); // URL for dubbed audio file

  const videoRef = React.useRef(null);
  const audioRef = React.useRef(null);
  const transcriptRef = React.useRef(null);
  const videoContainerRef = React.useRef(null);

  useEffect(() => {
    fetchInitialData();

    // Load subtitle preferences from localStorage
    const savedShowSubtitles = localStorage.getItem('showSubtitles');
    const savedSubtitleLanguage = localStorage.getItem('subtitleLanguage');

    if (savedShowSubtitles !== null) {
      setShowSubtitles(savedShowSubtitles === 'true');
    }
    if (savedSubtitleLanguage) {
      setSubtitleLanguage(savedSubtitleLanguage);
    }
  }, []);

  // Save subtitle preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem('showSubtitles', showSubtitles.toString());
  }, [showSubtitles]);

  useEffect(() => {
    localStorage.setItem('subtitleLanguage', subtitleLanguage);
  }, [subtitleLanguage]);

  // Auto-load existing dubbed files when lesson changes
  useEffect(() => {
    if (!currentLesson || currentLesson.type !== 'video') return;

    // Check if this lesson has pre-existing dubbed audio
    if (currentLesson.has_dubbed) {
      console.log('🎧 Auto-loading existing dubbed audio for:', currentLesson.title);

      // Build the path to the permanent dubbed audio
      const dubbedPath = currentLesson.path.replace(/\.[^.]+$/, '_dubbed.mp3');
      const fullAudioUrl = `http://localhost:3000/media/${dubbedPath}`;

      // Enable dubbing mode automatically
      setDubbedAudioUrl(fullAudioUrl);
      setUseDubbedAudio(true);
      setShowSubtitles(true);
      setSubtitleLanguage('vi');

      // Load Vietnamese subtitles
      loadSubtitles(true);
    } else {
      // Reset dubbing state for lessons without dubbed audio
      setUseDubbedAudio(false);
      setDubbedAudioUrl('');
    }
  }, [currentLesson]);

  const fetchInitialData = async () => {
    try {
      setLoading(true);
      const [coursesRes, statsRes, srtRes] = await Promise.all([
        axios.get(`${API_BASE}/courses`),
        axios.get(`${API_BASE}/stats`),
        axios.get(`${API_BASE}/videos-without-srt`)
      ]);
      setCourses(coursesRes.data.courses);
      setStats(statsRes.data);
      setVideosWithoutSrt(srtRes.data.videos);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  const selectFolder = async () => {
    try {
      const res = await axios.post(`${API_BASE}/select-folder`);
      if (res.data.status === 'success') {
        const scanRes = await axios.post(`${API_BASE}/scan-folder`, { folder_path: res.data.folder_path });
        if (scanRes.data.status === 'success') {
          fetchInitialData();
        }
      }
    } catch (error) {
      console.error("Folder selection failed:", error);
    }
  };

  const loadLessons = async (course) => {
    try {
      setSelectedCourse(course);
      const res = await axios.get(`${API_BASE}/lessons/${course.lessons_path}`);
      setLessons(res.data.lessons);
    } catch (error) {
      console.error("Error loading lessons:", error);
    }
  };

  const toggleProgress = async (lesson) => {
    try {
      const isCompleted = stats?.progress?.[lesson.path]?.completed;
      await axios.post(`${API_BASE}/update-progress`, {
        lesson_path: lesson.path,
        completed: !isCompleted
      });
      fetchInitialData();
    } catch (error) {
      console.error("Error updating progress:", error);
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;

    setIsCreatingFolder(true);
    try {
      await axios.post(`${API_BASE}/courses`, {
        name: newFolderName
      });

      alert(`Folder "${newFolderName}" created successfully!`);
      setNewFolderName('');
      fetchInitialData(); // Reload courses
      setActiveTab('courses'); // Switch back to courses
    } catch (error) {
      console.error("Create folder failed:", error);
      alert(`Failed to create folder: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsCreatingFolder(false);
    }
  };

  const runBatchTranscribe = async () => {
    try {
      setLoading(true);
      const res = await axios.post(`${API_BASE}/transcribe-all`);
      alert(`Batch results: ${res.data.transcribed.length} success, ${res.data.failed.length} failed.`);
      fetchInitialData();
    } catch (error) {
      console.error("Batch transcription failed:", error);
    } finally {
      setLoading(false);
    }
  };

  // Preload audio segment
  const preloadSegment = (index, url) => {
    const audio = new Audio(`http://localhost:3000${url}`);
    audio.preload = 'auto';
    setSegmentAudios(prev => ({ ...prev, [index]: audio }));
  };

  const handleDubbing = async (targetVideo) => {
    if (!targetVideo) return;

    // Only pause if we are currently watching the video being dubbed
    if (currentLesson && currentLesson.path === targetVideo.path && videoRef.current) {
      videoRef.current.pause();
    }

    // Set initial progress
    setDubbingProgressMap(prev => ({ ...prev, [targetVideo.path]: 0 }));
    console.log(`⏸️ Starting dubbing for: ${targetVideo.title}`);

    try {
      console.log('⚡ Creating dubbed audio...');

      // Call instant endpoint - returns after first 5 segments are ready
      const response = await axios.post('http://localhost:3000/api/dub-instant', {
        video_path: targetVideo.path,
        voice: 'vi-VN-HoaiMyNeural'
      });

      console.log('⚡ Response:', response.data);
      const { job_id, audio_url, total_segments, ready_segments, vi_srt_url } = response.data;

      if (!audio_url) {
        throw new Error('No audio generated');
      }

      const fullAudioUrl = `http://localhost:3000${audio_url}`;
      console.log(`✅ ${ready_segments} segments ready! Audio: ${fullAudioUrl}`);

      // Update progress
      const progress = Math.round((ready_segments / total_segments) * 100);
      setDubbingProgressMap(prev => ({ ...prev, [targetVideo.path]: progress }));

      // If we are currently watching this video, apply changes
      if (currentLesson && currentLesson.path === targetVideo.path) {
        setDubbedAudioUrl(fullAudioUrl);
        setShowSubtitles(true);
        setUseDubbedAudio(true);
        if (vi_srt_url) {
          setSubtitleLanguage('vi');
          setTimeout(() => loadSubtitles(true), 100);
        }
      }

      // STEP 4: Listen for full audio completion
      const eventSource = new EventSource(`http://localhost:3000/api/dub-instant-progress/${job_id}`);

      eventSource.addEventListener('complete', (event) => {
        const data = JSON.parse(event.data);
        console.log('🎉 Full audio ready:', data.full_audio_url);

        // Mark as 100% and remove from progress map (or keep it as 100)
        setDubbingProgressMap(prev => {
          const newMap = { ...prev };
          delete newMap[targetVideo.path]; // Remove spinner
          return newMap;
        });

        // Update local lesson state to show "Dubbed" status immediately
        fetchInitialData(); // Reload to update "has_dubbed" status in list

        // If watching, apply update
        if (currentLesson && currentLesson.path === targetVideo.path) {
          const fullUrl = `http://localhost:3000${data.full_audio_url}`;
          setDubbedAudioUrl(fullUrl);
          setSubtitleLanguage('vi');
          loadSubtitles();
        }

        eventSource.close();
      });

      // Listen for audio updates
      eventSource.addEventListener('audio_updated', (event) => {
        const data = JSON.parse(event.data);

        // Update progress in map
        setDubbingProgressMap(prev => ({ ...prev, [targetVideo.path]: data.progress }));

        // If watching, improved smart update logic
        if (currentLesson && currentLesson.path === targetVideo.path) {
          const newUrl = `http://localhost:3000${data.audio_url}?v=${data.version}`;
          pendingAudioUrlRef.current = newUrl;
        }
      });

      eventSource.addEventListener('error', () => {
        console.warn('SSE connection closed');
        eventSource.close();
      });

    } catch (error) {
      console.error('❌ Instant dubbing failed:', error);
      alert(`Không thể bắt đầu dubbing: ${error.message}`);
      setDubbingProgressMap(prev => {
        const newMap = { ...prev };
        delete newMap[targetVideo.path];
        return newMap;
      });
    }
  };

  // Enhanced Sync Logic for Dubbed Audio
  useEffect(() => {
    const video = videoRef.current;
    const audio = audioRef.current;
    if (!video || !audio) return;

    if (!useDubbedAudio) {
      // If dubbing disabled, ensure video is unmuted and audio is paused
      video.muted = false;
      audio.pause();
      return;
    }

    // Force mute video when dubbing is active
    video.muted = true;

    const handlePlay = () => {
      // Ensure video is muted
      video.muted = true;
      audio.currentTime = video.currentTime;
      audio.play().catch(e => {
        // Ignore AbortError which happens when pausing quickly
        if (e.name !== 'AbortError') console.warn('Audio play failed:', e);
      });
    };

    const handlePause = () => {
      audio.pause();
    };

    const handleSeeking = () => {
      audio.pause(); // Pause while seeking to prevent glitching
      audio.currentTime = video.currentTime;
    };

    const handleSeeked = () => {
      audio.currentTime = video.currentTime;
      if (!video.paused) {
        audio.play().catch(() => { });
      }
    };

    const handleTimeUpdate = () => {
      // ADAPTIVE SYNC (Variable Speed)
      // Instead of hard-seeking (which cuts audio), we adjust playback rate
      // to let audio "catch up" or "wait" smoothly.
      if (!video.paused && !audio.paused) {
        const drift = audio.currentTime - video.currentTime;

        if (Math.abs(drift) > 3.0) {
          // Hard sync only for massive lag (>3s)
          console.log(`⚠️ Massive drift (${drift.toFixed(2)}s) -> Hard Reset`);
          audio.currentTime = video.currentTime;
          audio.playbackRate = 1.0;
        } else if (drift < -0.1) {
          // Audio is lagging behind video -> Speed up slightly
          // Lag -0.1s to -3.0s
          // Formula: 1.0 + (drift * 0.1) -> e.g. drift -1s = 1.1x speed
          audio.playbackRate = Math.min(1.0 + Math.abs(drift) * 0.1, 1.25);
        } else if (drift > 0.1) {
          // Audio is ahead of video -> Slow down slightly
          // Ahead 0.1s to 3.0s
          audio.playbackRate = Math.max(1.0 - (drift * 0.1), 0.75);
        } else {
          // Perfect sync (within 0.1s)
          audio.playbackRate = 1.0;
        }
      }

      // SMART BUFFER CHECK
      // If we have a pending update and we are near the end of the current audio file (buffer < 15s)
      if (pendingAudioUrlRef.current && (audio.duration - audio.currentTime < 15)) {
        console.log('🔄 Buffer low (<15s), applying pending audio update...');
        const newUrl = pendingAudioUrlRef.current;
        pendingAudioUrlRef.current = null; // Clear pending

        // Store restore position for onLoadedData callback
        restorePositionRef.current = audio.currentTime;

        // Set new URL - will trigger onLoadedData which will restore position
        setDubbedAudioUrl(newUrl);
      }
    };


    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('seeking', handleSeeking);
    video.addEventListener('seeked', handleSeeked);
    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('seeking', handleSeeking);
      video.removeEventListener('seeked', handleSeeked);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      // Stop audio on cleanup
      audio.pause();
    };
  }, [useDubbedAudio, currentLesson]);

  const handleSingleTranscribe = async () => {
    if (!currentLesson) return;
    try {
      setProcessingSTT(true);
      const res = await axios.post(`${API_BASE}/transcribe`, { video_path: currentLesson.path });
      alert("Transcription complete!");
      fetchInitialData();
      // Reload lessons to get the new subtitles
      loadLessons(selectedCourse);
    } catch (error) {
      alert(error.response?.data?.detail || "Transcription failed");
    } finally {
      setProcessingSTT(false);
    }
  };

  const toggleFullscreen = () => {
    const container = videoContainerRef.current;
    if (!container) return;

    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };


  // Reset dubbed audio state when changing lessons
  useEffect(() => {
    setUseDubbedAudio(false);
    loadSubtitles();
  }, [currentLesson]);

  // Track video time for subtitle highlighting
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    return () => video.removeEventListener('timeupdate', handleTimeUpdate);
  }, [currentLesson]);

  // Force enable Vietnamese subtitles when available
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !currentLesson || !currentLesson.has_dubbed) return;

    const enableVietnameseSubtitles = () => {
      if (video.textTracks && video.textTracks.length > 0) {
        for (let i = 0; i < video.textTracks.length; i++) {
          const track = video.textTracks[i];
          if (track.language === 'vi') {
            track.mode = 'showing';
            console.log('✅ Enabled Vietnamese subtitles');
          } else {
            track.mode = 'disabled';
          }
        }
      }
    };

    // Try immediately
    enableVietnameseSubtitles();

    // Also try after a short delay to ensure tracks are loaded
    const timer = setTimeout(enableVietnameseSubtitles, 500);

    // Listen for track changes
    const handleLoadedMetadata = () => enableVietnameseSubtitles();
    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      clearTimeout(timer);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [currentLesson]);

  // Track fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      const isNowFullscreen = !!(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement
      );
      setIsFullscreen(isNowFullscreen);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);

  const parseSrtTime = (timeStr) => {
    // Parse both 00:00:01,850 and 00:00:01.850 formats
    // Replace comma with dot for consistent parsing
    const normalized = timeStr.trim().replace(',', '.');
    const parts = normalized.split(':');

    if (parts.length === 3) {
      const hours = parseInt(parts[0]) || 0;
      const minutes = parseInt(parts[1]) || 0;
      const seconds = parseFloat(parts[2]) || 0;
      return hours * 3600 + minutes * 60 + seconds;
    }

    console.warn('Invalid SRT timestamp:', timeStr);
    return 0;
  };

  const loadSubtitles = async (forceLoadVi = false) => {
    if (!currentLesson || currentLesson.type !== 'video') {
      setSubtitles([]);
      setVietnameseSubtitles([]);
      console.log('❌ No video lesson selected');
      return;
    }

    // Even if no original EN subtitle, still try to load VI if dubbing is active
    const hasOriginalSubtitle = currentLesson.has_subtitle;
    const shouldLoadVi = forceLoadVi || useDubbedAudio;
    console.log('🔍 loadSubtitles called: forceLoadVi=', forceLoadVi, 'useDubbedAudio=', useDubbedAudio, 'shouldLoadVi=', shouldLoadVi);

    try {
      // Load English subtitles (only if original subtitle exists)
      if (hasOriginalSubtitle) {
        const srtPath = currentLesson.path.replace(/\.[^.]+$/, '.srt');
        const response = await fetch(`http://localhost:3000/media/${srtPath}`);
        if (response.ok) {
          const text = await response.text();

          console.log('📝 Loading EN subtitles from:', srtPath);

          // Parse SRT - normalize line endings first (handle Windows \r\n)
          const normalizedText = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
          const blocks = normalizedText.split('\n\n').filter(b => b.trim());
          const parsed = [];

          for (const block of blocks) {
            const lines = block.split('\n');
            if (lines.length >= 3) {
              const timeLine = lines[1];
              if (timeLine.includes('-->')) {
                const [start, end] = timeLine.split('-->').map(t => t.trim());
                const subtitle = {
                  start: parseSrtTime(start),
                  end: parseSrtTime(end),
                  text: lines.slice(2).join(' ')
                };
                parsed.push(subtitle);
              }
            }
          }

          setSubtitles(parsed);
          console.log(`✅ Loaded ${parsed.length} EN subtitles`);
        }
      }

      // Load Vietnamese subtitles - always try when dubbing is active
      if (currentLesson.has_dubbed || shouldLoadVi) {
        try {
          const viSrtPath = currentLesson.path.replace(/\.[^.]+$/, '_vi.srt');
          console.log('📝 Loading VI subtitles from:', viSrtPath);

          const viResponse = await fetch(`http://localhost:3000/media/${viSrtPath}`);

          if (!viResponse.ok) {
            throw new Error(`HTTP ${viResponse.status}: ${viResponse.statusText}`);
          }

          const viText = await viResponse.text();
          console.log('📝 VI SRT file loaded, length:', viText.length);

          // Normalize line endings for VI subtitles too
          const normalizedViText = viText.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
          const viBlocks = normalizedViText.split('\n\n').filter(b => b.trim());
          const viParsed = [];

          for (const block of viBlocks) {
            const lines = block.split('\n');
            if (lines.length >= 3) {
              const timeLine = lines[1];
              if (timeLine.includes('-->')) {
                const [start, end] = timeLine.split('-->').map(t => t.trim());
                const subtitle = {
                  start: parseSrtTime(start),
                  end: parseSrtTime(end),
                  text: lines.slice(2).join(' ')
                };
                viParsed.push(subtitle);
              }
            }
          }

          setVietnameseSubtitles(viParsed);
          setSubtitleLanguage('vi'); // Default to Vietnamese if available
          console.log(`✅ Loaded ${viParsed.length} VI subtitles, language set to VI`);
        } catch (error) {
          console.error('❌ Failed to load Vietnamese subtitles:', error);
          setVietnameseSubtitles([]);
          setSubtitleLanguage('en');
        }
      } else {
        console.log('⚠️ No dubbing available, VI subtitles not loaded');
        setVietnameseSubtitles([]);
        setSubtitleLanguage('en');
      }
    } catch (error) {
      console.error('❌ Failed to load subtitles:', error);
      setSubtitles([]);
      setVietnameseSubtitles([]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-blue-950 text-white">
      {/* Sidebar */}
      <div className="fixed left-0 top-0 h-screen w-64 glass-panel-solid p-6 flex flex-col z-50">
        <div className="p-6">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-2">
            <PlayCircle className="text-blue-500" /> AI HUB
          </h1>
        </div>

        <nav className="flex-1 px-4 space-y-2">
          <TabButton
            active={activeTab === 'dashboard'}
            onClick={() => setActiveTab('dashboard')}
            icon={<TrendingUp size={20} />}
            label="Dashboard"
          />
          <TabButton
            active={activeTab === 'courses'}
            onClick={() => setActiveTab('courses')}
            icon={<Library size={20} />}
            label="Courses"
          />
          <TabButton
            active={activeTab === 'stt'}
            onClick={() => setActiveTab('stt')}
            icon={<Mic size={20} />}
            label="Speech to Text"
          />
          <TabButton
            active={activeTab === 'dubbing'}
            onClick={() => setActiveTab('dubbing')}
            icon={<Languages size={20} />}
            label="AI Dubbing"
          />
          <TabButton
            active={activeTab === 'progress'}
            onClick={() => setActiveTab('progress')}
            icon={<CheckCircle2 size={20} />}
            label="Learning Progress"
          />
        </nav>

        <div className="p-4 border-t border-white/5">
          <button
            onClick={() => setActiveTab('add-folder')}
            className="w-full glass-panel py-3 px-4 flex items-center justify-center gap-2 hover:bg-white/5 transition-all"
          >
            <FolderPlus size={20} className="text-blue-400" />
            <span className="font-medium">Add Folder</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="ml-64 h-screen p-8 overflow-y-auto premium-scroll">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-2xl font-bold capitalize">{activeTab}</h2>
            <p className="text-slate-400 text-sm">Welcome back to your learning journey</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="px-4 py-2 glass-panel flex items-center gap-2 text-sm">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              Backend Active
            </div>
            <button className="p-2 glass-panel hover:bg-white/5 transition-colors">
              <Settings size={20} className="text-slate-400" />
            </button>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {activeTab === 'courses' && (
            <motion.div
              key="courses"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {courses.map(course => (
                <div
                  key={course.id}
                  className="glass-panel card group cursor-pointer"
                  onClick={() => {
                    loadLessons(course);
                    setActiveTab('lessons');
                  }}
                >
                  <div className="flex justify-between items-start mb-4">
                    <div className="p-3 bg-blue-500/10 rounded-xl text-blue-400 group-hover:scale-110 transition-transform">
                      <Library size={24} />
                    </div>
                    <span className="badge badge-video">{course.video_count} Videos</span>
                  </div>
                  <h3 className="text-lg font-bold mb-1">{course.name}</h3>
                  <p className="text-slate-500 text-sm mb-4 truncate">{course.path}</p>

                  <div className="mt-4 pt-4 border-t border-white/5 flex justify-between items-center">
                    <span className="text-xs text-slate-400">Total: {course.total_lessons} items</span>
                    <PlayCircle className="text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity" size={20} />
                  </div>
                </div>
              ))}
            </motion.div>
          )}

          {activeTab === 'lessons' && (
            <motion.div
              key="lessons"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="h-full flex flex-col"
            >
              <div className="flex justify-between items-center mb-6">
                <div>
                  <button
                    onClick={() => setActiveTab('courses')}
                    className="text-blue-400 hover:text-blue-300 flex items-center gap-2 mb-2 px-0"
                  >
                    &larr; Back to Library
                  </button>
                  <h3 className="text-xl font-bold">{currentLesson?.title || selectedCourse?.name}</h3>
                </div>
              </div>

              <div className="study-container">
                {/* Center: Video Player */}
                <div
                  ref={videoContainerRef}
                  className="video-section glass-panel"
                  style={{
                    position: 'relative',
                    background: isFullscreen ? '#000' : undefined
                  }}
                  onDoubleClick={() => {
                    // Fullscreen the container (not just video) so subtitles are included
                    const container = videoContainerRef.current;
                    if (container) {
                      if (document.fullscreenElement) {
                        document.exitFullscreen();
                      } else {
                        container.requestFullscreen?.() ||
                          container.webkitRequestFullscreen?.() ||
                          container.msRequestFullscreen?.();
                      }
                    }
                  }}
                >
                  {currentLesson && currentLesson.type === 'video' ? (
                    <>
                      <video
                        ref={videoRef}
                        controls
                        autoPlay
                        muted={useDubbedAudio}
                        key={currentLesson.path}
                        src={`http://localhost:3000/media/${currentLesson.path}`}
                        className="w-full h-full"
                        crossOrigin="anonymous"
                        style={{ maxHeight: isFullscreen ? '100vh' : undefined }}
                      />

                      {/* Custom Subtitle Overlay (YouTube style) - Works in fullscreen */}
                      {showSubtitles && (subtitles.length > 0 || vietnameseSubtitles.length > 0) && (() => {
                        const currentSubs = subtitleLanguage === 'vi' ? vietnameseSubtitles : subtitles;
                        const currentSub = currentSubs.find(sub => currentTime >= sub.start && currentTime <= sub.end);

                        return currentSub ? (
                          <div style={{
                            position: 'absolute',
                            bottom: isFullscreen ? '100px' : '60px',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            background: 'rgba(0,0,0,0.85)',
                            color: 'white',
                            padding: isFullscreen ? '16px 28px' : '8px 16px',
                            borderRadius: '6px',
                            fontSize: isFullscreen ? '28px' : '18px',
                            fontWeight: '500',
                            maxWidth: '85%',
                            textAlign: 'center',
                            lineHeight: '1.5',
                            pointerEvents: 'none',
                            zIndex: 99999,
                            textShadow: '2px 2px 4px rgba(0,0,0,0.8)'
                          }}>
                            {currentSub.text}
                          </div>
                        ) : null;
                      })()}

                      {/* Subtitle Controls */}
                      {(subtitles.length > 0 || vietnameseSubtitles.length > 0) && (
                        <div style={{
                          position: 'absolute',
                          bottom: isFullscreen ? '24px' : '16px',
                          right: isFullscreen ? '24px' : '16px',
                          display: 'flex',
                          gap: '8px',
                          zIndex: 99998
                        }}>
                          {/* CC Toggle */}
                          <button
                            onClick={() => setShowSubtitles(!showSubtitles)}
                            title={showSubtitles ? "Tắt phụ đề" : "Bật phụ đề"}
                            style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '4px',
                              border: 'none',
                              background: showSubtitles ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.6)',
                              color: showSubtitles ? '#000' : '#fff',
                              cursor: 'pointer',
                              fontSize: '14px',
                              fontWeight: '600',
                              transition: 'all 0.2s'
                            }}
                          >
                            CC
                          </button>

                          {/* Language Switch (only if Vietnamese available) */}
                          {vietnameseSubtitles.length > 0 && showSubtitles && (
                            <button
                              onClick={() => setSubtitleLanguage(subtitleLanguage === 'en' ? 'vi' : 'en')}
                              title={subtitleLanguage === 'en' ? "Chuyển sang tiếng Việt" : "Switch to English"}
                              style={{
                                padding: '8px 12px',
                                borderRadius: '4px',
                                border: 'none',
                                background: 'rgba(255,255,255,0.9)',
                                color: '#000',
                                cursor: 'pointer',
                                fontSize: '13px',
                                fontWeight: '600',
                                transition: 'all 0.2s'
                              }}
                            >
                              {subtitleLanguage === 'en' ? 'EN' : 'VI'}
                            </button>
                          )}
                          {vietnameseSubtitles.length === 0 && showSubtitles && (
                            <button
                              onClick={() => {
                                alert('Vui lòng tạo lồng tiếng tiếng Việt trước (nút Languages)');
                              }}
                              title="Chuất tạo lồng tiếng tiếng Việt"
                              style={{
                                padding: '8px 12px',
                                borderRadius: '4px',
                                border: 'none',
                                background: 'rgba(128,128,128,0.6)',
                                color: '#fff',
                                cursor: 'pointer',
                                fontSize: '13px',
                                fontWeight: '600',
                                opacity: 0.7
                              }}
                            >
                              VI
                            </button>
                          )}
                          {/* Fullscreen Toggle */}
                          <button
                            onClick={toggleFullscreen}
                            title="Toàn màn hình"
                            style={{
                              background: 'none',
                              border: 'none',
                              color: 'white',
                              cursor: 'pointer',
                              padding: '4px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              opacity: 0.8,
                              transition: 'opacity 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
                            onMouseLeave={(e) => e.currentTarget.style.opacity = '0.8'}
                          >
                            {isFullscreen ? (
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
                              </svg>
                            ) : (
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
                              </svg>
                            )}
                          </button>
                        </div>
                      )}

                      <style jsx>{`
                        /* Hide native fullscreen button to force use of our custom one */
                        video::-webkit-media-controls-fullscreen-button {
                          display: none !important;
                        }
                        video::-webkit-media-controls-enclosure {
                          z-index: 1;
                        }
                      `}</style>

                      {/* Hidden audio element for dubbed audio */}
                      {useDubbedAudio && dubbedAudioUrl && (
                        <audio
                          ref={audioRef}
                          src={dubbedAudioUrl}
                          preload="auto"
                          style={{ display: 'none' }}
                          onLoadedData={() => {
                            console.log('🔊 Audio loaded:', dubbedAudioUrl);

                            // Seamless position restore after Smart Update
                            if (restorePositionRef.current !== null) {
                              const pos = restorePositionRef.current;
                              restorePositionRef.current = null;

                              if (audioRef.current) {
                                audioRef.current.currentTime = pos;
                                // Resume playback if video is playing
                                if (videoRef.current && !videoRef.current.paused) {
                                  audioRef.current.play().catch(() => { });
                                }
                                console.log('✅ Audio position restored to:', pos);
                              }
                            }
                          }}
                          onCanPlay={() => {
                            if (audioRef.current) {
                              audioRef.current.volume = 1.0;
                            }
                          }}
                          onError={(e) => console.error('❌ Audio error:', e.target.error)}
                        />
                      )}


                      {/* Control Buttons - Always Visible */}
                      <div style={{
                        position: 'absolute',
                        top: '16px',
                        right: '16px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '12px',
                        zIndex: 10
                      }}>
                        {/* Dubbing Toggle Button */}
                        <button
                          onClick={() => {
                            if (currentLesson.has_dubbed) {
                              setUseDubbedAudio(!useDubbedAudio);
                            } else if (!currentLesson.has_subtitle) {
                              alert("Vui lòng tạo phụ đề trước (nút Microphone)");
                            } else {
                              handleDubbing();
                            }
                          }}
                          disabled={processingDub}
                          title={
                            currentLesson.has_dubbed
                              ? (useDubbedAudio ? "Tắt lồng tiếng" : "Bật lồng tiếng")
                              : (!currentLesson.has_subtitle ? "Cần phụ đề trước" : "Tạo lồng tiếng AI")
                          }
                          style={{
                            width: '48px',
                            height: '48px',
                            borderRadius: '50%',
                            border: 'none',
                            cursor: processingDub || !currentLesson.has_subtitle && !currentLesson.has_dubbed ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: processingDub
                              ? 'rgb(59, 130, 246)'
                              : currentLesson.has_dubbed
                                ? (useDubbedAudio ? 'rgb(34, 197, 94)' : 'rgb(37, 99, 235)')
                                : (!currentLesson.has_subtitle ? 'rgb(71, 85, 105)' : 'rgb(37, 99, 235)'),
                            color: 'white',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                            transition: 'all 0.2s',
                            opacity: !currentLesson.has_subtitle && !currentLesson.has_dubbed ? 0.5 : 1
                          }}
                        >
                          {processingDub ? (
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                              <RotateCw size={20} />
                            </motion.div>
                          ) : (
                            <Languages size={20} />
                          )}
                        </button>

                        {/* Transcription Button */}
                        <button
                          onClick={handleSingleTranscribe}
                          disabled={processingSTT || currentLesson.has_subtitle}
                          title={currentLesson.has_subtitle ? "Đã có phụ đề" : "Tạo phụ đề"}
                          style={{
                            width: '48px',
                            height: '48px',
                            borderRadius: '50%',
                            border: 'none',
                            cursor: processingSTT || currentLesson.has_subtitle ? 'not-allowed' : 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: processingSTT
                              ? 'rgb(245, 158, 11)'
                              : currentLesson.has_subtitle
                                ? 'rgb(34, 197, 94)'
                                : 'rgb(245, 158, 11)',
                            color: 'white',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                            transition: 'all 0.2s',
                            opacity: currentLesson.has_subtitle ? 0.7 : 1
                          }}
                        >
                          {processingSTT ? (
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                              <RotateCw size={20} />
                            </motion.div>
                          ) : (
                            <Mic size={20} />
                          )}
                        </button>
                      </div>
                    </>
                  ) : currentLesson && currentLesson.type === 'pdf' ? (
                    <iframe
                      src={`http://localhost:3000/media/${currentLesson.path}`}
                      className="w-full h-full border-0"
                      title="PDF Viewer"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-400">
                      <p>Select a lesson to start</p>
                    </div>
                  )}
                </div>

                {/* Right: Lesson Playlist */}
                <div className="lesson-list-sidebar overflow-y-auto premium-scroll px-2">
                  <h4 className="text-sm uppercase tracking-widest text-slate-500 font-bold mb-2">🎬 Playlist</h4>
                  <div className="flex flex-col gap-2">
                    {lessons
                      .filter(l => l.type !== 'subtitle')
                      .map((lesson, index) => {
                        const isCompleted = stats?.progress?.[lesson.path]?.completed;
                        const isActive = currentLesson?.path === lesson.path;

                        return (
                          <div
                            key={lesson.path}
                            className={`lesson-item ${isActive ? 'active' : ''} flex gap-3 p-3 rounded-xl cursor-pointer transition-colors ${isActive ? 'bg-blue-500/20 border border-blue-500/50' : 'bg-slate-800/50 hover:bg-white/5 border border-white/5'}`}
                            onClick={() => {
                              if (lesson.type === 'video') {
                                setCurrentLesson(lesson);
                              } else {
                                window.open(`http://localhost:3000/media/${lesson.path}`, '_blank');
                              }
                            }}
                          >
                            {/* Video Number */}
                            <div className={`mt-0.5 shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${isActive ? 'bg-blue-500 text-white' : 'bg-slate-700 text-slate-400'}`}>
                              {index + 1}
                            </div>

                            <div className="flex-1 min-w-0 flex flex-col justify-center">
                              <h5
                                className={`text-sm font-medium leading-tight line-clamp-2 break-words ${isCompleted ? 'text-slate-500' : isActive ? 'text-white' : 'text-slate-300'}`}
                                title={lesson.title}
                              >
                                {lesson.title}
                              </h5>
                              <div className="flex items-center gap-2 mt-2">
                                {isCompleted && <CheckCircle2 size={12} className="text-green-500" />}
                                <span className="text-[10px] text-slate-500 uppercase tracking-wider">{lesson.type}</span>
                                {lesson.type === 'video' && (
                                  <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${lesson.has_subtitle
                                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                                    : 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                                    }`}>
                                    {lesson.has_subtitle ? 'SUB' : 'NO SUB'}
                                  </span>
                                )}
                              </div>
                            </div>

                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleProgress(lesson);
                              }}
                              className={`shrink-0 self-center btn-check ${isCompleted ? 'completed' : ''}`}
                              title={isCompleted ? "Mark as incomplete" : "Mark as completed"}
                            >
                              <CheckCircle2 size={16} strokeWidth={3} />
                            </button>
                          </div>
                        );
                      })}
                  </div>
                </div>

                {/* Transcript Panel - Below Video */}
                <div className="glass-panel" style={{
                  marginTop: '16px',
                  maxHeight: '300px',
                  overflowY: 'auto',
                  padding: '16px'
                }}>
                  {currentLesson && currentLesson.type === 'video' && (subtitles.length > 0 || vietnameseSubtitles.length > 0) ? (() => {
                    const transcriptSubs = vietnameseSubtitles.length > 0 ? vietnameseSubtitles : subtitles;

                    return (
                      <div className="h-full flex flex-col">
                        <h3 style={{
                          fontSize: '14px',
                          fontWeight: '600',
                          marginBottom: '12px',
                          color: 'rgba(255,255,255,0.9)'
                        }}>📝 Transcript ({vietnameseSubtitles.length > 0 ? 'VI' : 'EN'})</h3>
                        <div ref={transcriptRef} className="flex-1 overflow-y-auto premium-scroll pr-2" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          {transcriptSubs.map((sub, idx) => {
                            const isActive = currentTime >= sub.start && currentTime <= sub.end;

                            // Auto-scroll logic here might need adjustment for sidebar height context
                            if (isActive && transcriptRef.current) {
                              const activeElement = transcriptRef.current.children[idx];
                              if (activeElement) {
                                // Use scrollIntoView but careful not to jerk the whole page
                                activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                              }
                            }

                            return (
                              <div
                                key={idx}
                                onClick={() => {
                                  if (videoRef.current) {
                                    videoRef.current.currentTime = sub.start;
                                  }
                                }}
                                style={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: '4px',
                                  padding: '10px',
                                  borderRadius: '8px',
                                  cursor: 'pointer',
                                  background: isActive
                                    ? 'rgba(59, 130, 246, 0.15)'
                                    : 'rgba(255,255,255,0.02)',
                                  border: isActive
                                    ? '1px solid rgba(59, 130, 246, 0.3)'
                                    : '1px solid transparent',
                                  transition: 'all 0.2s'
                                }}
                              >
                                <span style={{
                                  fontSize: '11px',
                                  color: isActive ? '#60a5fa' : '#64748b',
                                  fontWeight: '600'
                                }}>
                                  {Math.floor(sub.start / 60)}:{String(Math.floor(sub.start % 60)).padStart(2, '0')}
                                </span>
                                <p style={{
                                  fontSize: '13px',
                                  color: isActive ? '#fff' : '#cbd5e1',
                                  margin: 0,
                                  lineHeight: '1.4'
                                }}>
                                  {sub.text}
                                </p>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })() : (
                    <div className="flex items-center justify-center h-full text-slate-500 italic p-4 text-center">
                      <p>No transcript available</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'stt' && (
            <motion.div
              key="stt"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div className="glass-panel p-6 border-l-4 border-amber-500 flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    <AlertCircle className="text-amber-500" /> Transcribe Workspace
                  </h3>
                  <p className="text-slate-400 mt-1">Found {videosWithoutSrt.length} videos needing subtitles</p>
                </div>
                <button
                  onClick={runBatchTranscribe}
                  className="btn btn-primary"
                  disabled={loading || videosWithoutSrt.length === 0}
                >
                  {loading ? 'Processing...' : 'Start Batch Process'}
                </button>
              </div>

              <div className="grid gap-4">
                {videosWithoutSrt.map(video => (
                  <div key={video.path} className="glass-panel p-4 flex items-center justify-between group">
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-slate-500/10 rounded-lg text-slate-500 group-hover:text-amber-500 transition-colors">
                        <Mic size={20} />
                      </div>
                      <div>
                        <h4 className="font-medium">{video.name}</h4>
                        <p className="text-xs text-slate-500">{video.course}</p>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button className="btn btn-ghost text-xs">Preview</button>
                      <button className="btn btn-primary text-xs px-4">Transcribe</button>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === 'dubbing' && (
            <motion.div
              key="dubbing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div className="glass-panel p-6 border-l-4 border-blue-500 flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-bold flex items-center gap-2">
                    <Languages className="text-blue-500" /> AI Dubbing Center
                  </h3>
                  <p className="text-slate-400 mt-1">Select videos to generate Vietnamese dubbing</p>
                </div>
                <button onClick={fetchInitialData} className="btn btn-primary">
                  <RotateCw size={16} className="mr-2" /> Refresh
                </button>
              </div>

              {/* Show instruction if no course selected */}
              {!selectedCourse ? (
                <div className="glass-panel p-8 text-center">
                  <Languages size={48} className="mx-auto text-slate-600 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Select a Course First</h3>
                  <p className="text-slate-400 mb-4">Go to Courses tab and open a course to see videos ready for dubbing</p>
                  <button onClick={() => setActiveTab('courses')} className="btn btn-primary">
                    Browse Courses
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {lessons.filter(l => l.has_subtitle && l.type === 'video').map(video => (
                    <div key={video.path} className="glass-panel p-4 flex flex-col gap-4">
                      <div className="flex items-center gap-4">
                        <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
                          <PlayCircle size={20} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium truncate">{video.title}</h4>
                          <div className="flex gap-2 mt-1">
                            <span className="badge badge-srt">Has Subtitle</span>
                            {video.has_dubbed && <span className="badge badge-dubbed">Already Dubbed</span>}
                          </div>
                        </div>
                      </div>
                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => {
                            setCurrentLesson(video);
                            setActiveTab('lessons');
                          }}
                          className="btn btn-ghost flex-1 text-xs justify-center"
                        >
                          Preview
                        </button>
                        <button
                          onClick={() => {
                            if (!video.has_dubbed && !dubbingProgressMap[video.path]) {
                              handleDubbing(video);
                            }
                          }}
                          disabled={video.has_dubbed || dubbingProgressMap[video.path] !== undefined}
                          className="btn btn-primary text-xs min-w-[100px]"
                        >
                          {video.has_dubbed ? (
                            'Dubbed ✓'
                          ) : dubbingProgressMap[video.path] !== undefined ? (
                            <div className="flex items-center gap-2">
                              <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                              {Math.round(dubbingProgressMap[video.path])}%
                            </div>
                          ) : (
                            'Dub Now'
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                  {lessons.filter(l => l.has_subtitle && l.type === 'video').length === 0 && (
                    <div className="col-span-full p-12 text-center text-slate-500 italic">
                      <AlertCircle size={32} className="mx-auto mb-3 text-slate-600" />
                      <p>No videos with subtitles found in {selectedCourse.name}</p>
                      <p className="text-sm mt-2">Transcribe videos first in the STT tab</p>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'dashboard' && stats && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
              <StatCard
                label="Overall Progress"
                value={`${stats.overall_percentage}%`}
                icon={<CheckCircle2 className="text-green-500" />}
                color="text-green-500"
              />
              <StatCard
                label="Videos Completed"
                value={`${stats.completed_videos} / ${stats.total_videos}`}
                icon={<PlayCircle className="text-blue-500" />}
                color="text-blue-500"
              />
              <StatCard
                label="Exercises Finished"
                value={`${stats.completed_exercises} / ${stats.total_exercises}`}
                icon={<FileText className="text-purple-500" />}
                color="text-purple-500"
              />
              <StatCard
                label="Missing SRT"
                value={videosWithoutSrt.length}
                icon={<AlertCircle className="text-amber-500" />}
                color="text-amber-500"
              />
            </motion.div>
          )}

          {activeTab === 'add-folder' && (
            <motion.div
              key="add-folder"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="max-w-md mx-auto mt-20"
            >
              <div className="glass-panel p-8">
                <div className="flex items-center gap-3 mb-6 text-blue-400">
                  <div className="p-3 bg-blue-500/10 rounded-xl">
                    <FolderPlus size={32} />
                  </div>
                  <h3 className="text-xl font-bold text-white">Create New Course Folder</h3>
                </div>

                <p className="text-slate-400 mb-6">
                  Add a new folder to your library to organize your videos and materials.
                </p>

                <div className="space-y-4">
                  <div>
                    <label className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-2 block">
                      Folder Name
                    </label>
                    <input
                      type="text"
                      value={newFolderName}
                      onChange={(e) => setNewFolderName(e.target.value)}
                      placeholder="e.g. Python_Advanced_Course"
                      className="w-full bg-slate-900/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-blue-500 transition-colors"
                      onKeyDown={(e) => e.key === 'Enter' && handleCreateFolder()}
                    />
                  </div>

                  <button
                    onClick={handleCreateFolder}
                    disabled={isCreatingFolder || !newFolderName.trim()}
                    className="w-full btn btn-primary py-3 justify-center text-base"
                  >
                    {isCreatingFolder ? (
                      <>
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ repeat: Infinity, duration: 1 }}
                          className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                        />
                        Creating...
                      </>
                    ) : (
                      <>
                        <FolderPlus size={20} />
                        Create Folder
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => setActiveTab('courses')}
                    className="w-full btn btn-ghost py-2 justify-center text-sm text-slate-500"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function TabButton({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl nav-tab ${active ? 'active' : 'text-slate-400'}`}
    >
      {icon}
      <span className="font-medium">{label}</span>
      {active && <motion.div layoutId="activeTab" className="ml-auto w-1 h-4 bg-blue-500 rounded-full" />}
    </button>
  );
}

function StatCard({ label, value, icon, color }) {
  return (
    <div className="glass-panel p-6 flex flex-col items-center text-center">
      <div className="p-4 bg-white/5 rounded-2xl mb-4">
        {React.cloneElement(icon, { size: 32 })}
      </div>
      <p className="text-slate-400 text-sm mb-1 uppercase tracking-wider">{label}</p>
      <h4 className={`text-3xl font-bold ${color}`}>{value}</h4>
    </div>
  );
}

export default App;
