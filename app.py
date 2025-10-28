import os
import sys

# Set environment variables BEFORE importing streamlit
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/.streamlit'
os.environ['HOME'] = '/tmp'

import streamlit as st
import requests
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
import tempfile
warnings.filterwarnings('ignore')

# Voice Emotion Detector Class
class VoiceEmotionDetector:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

    def extract_features(self):
        """Extract comprehensive audio features"""
        features = {}

        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)[0]
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(self.y)[0]

        # Energy and amplitude
        features['rms'] = librosa.feature.rms(y=self.y)[0]

        # Pitch features
        features['pitch'] = librosa.yin(self.y, fmin=50, fmax=500)

        # MFCCs (Mel-frequency cepstral coefficients)
        features['mfccs'] = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)

        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(y=self.y, sr=self.sr)

        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        # Convert tempo to scalar if it's an array
        features['tempo'] = float(tempo) if isinstance(tempo, np.ndarray) else tempo

        return features

    def detect_emotion(self, features):
        """Rule-based emotion detection using audio features"""
        # Calculate mean values
        mean_pitch = np.mean(features['pitch'][~np.isnan(features['pitch'])])
        mean_energy = np.mean(features['rms'])
        mean_zcr = np.mean(features['zero_crossing_rate'])
        mean_spectral_centroid = np.mean(features['spectral_centroid'])
        tempo = float(features['tempo']) if isinstance(features['tempo'], np.ndarray) else features['tempo']

        # Emotion detection logic
        emotions_score = {
            'Happy': 0,
            'Sad': 0,
            'Angry': 0,
            'Neutral': 0,
            'Fearful': 0,
            'Surprised': 0
        }

        # Happy: High pitch, high energy, fast tempo
        if mean_pitch > 180 and mean_energy > 0.05 and tempo > 120:
            emotions_score['Happy'] += 3

        # Sad: Low pitch, low energy, slow tempo
        if mean_pitch < 150 and mean_energy < 0.03 and tempo < 90:
            emotions_score['Sad'] += 3

        # Angry: High energy, high ZCR, moderate to high pitch
        if mean_energy > 0.06 and mean_zcr > 0.08:
            emotions_score['Angry'] += 3

        # Neutral: Moderate values
        if 140 < mean_pitch < 180 and 0.02 < mean_energy < 0.05:
            emotions_score['Neutral'] += 2

        # Fearful: High pitch, variable energy, high ZCR
        if mean_pitch > 190 and mean_zcr > 0.09:
            emotions_score['Fearful'] += 2

        # Surprised: Sudden energy changes, high pitch
        if mean_pitch > 185 and np.std(features['rms']) > 0.02:
            emotions_score['Surprised'] += 2

        # Normalize scores
        total = sum(emotions_score.values())
        if total > 0:
            emotions_score = {k: (v/total)*100 for k, v in emotions_score.items()}
        else:
            # If no emotion detected, set Neutral as default
            emotions_score['Neutral'] = 100

        detected_emotion = max(emotions_score, key=emotions_score.get)

        return detected_emotion, emotions_score

    def visualize_all(self):
        """Create comprehensive visualizations"""
        features = self.extract_features()
        emotion, emotion_scores = self.detect_emotion(features)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))

        # 1. Waveform
        plt.subplot(4, 3, 1)
        time = np.linspace(0, self.duration, len(self.y))
        plt.plot(time, self.y, color='#2E86AB', linewidth=0.5)
        plt.title('Audio Waveform (Amplitude vs Time)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        # 2. Frequency Spectrum (FFT)
        plt.subplot(4, 3, 2)
        n = len(self.y)
        fft_vals = fft(self.y)
        fft_freq = fftfreq(n, 1/self.sr)
        positive_freq_idx = np.where(fft_freq >= 0)
        plt.plot(fft_freq[positive_freq_idx][:n//2],
                 np.abs(fft_vals[positive_freq_idx][:n//2]),
                 color='#A23B72', linewidth=0.5)
        plt.title('Frequency Spectrum (FFT)', fontsize=12, fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 5000)
        plt.grid(True, alpha=0.3)

        # 3. Spectrogram
        plt.subplot(4, 3, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram (Frequency vs Time)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency (Hz)')

        # 4. Mel Spectrogram
        plt.subplot(4, 3, 4)
        mel_spec = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=self.sr, x_axis='time', y_axis='mel', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram', fontsize=12, fontweight='bold')

        # 5. Spectral Centroid
        plt.subplot(4, 3, 5)
        frames = range(len(features['spectral_centroid']))
        t = librosa.frames_to_time(frames, sr=self.sr)
        plt.plot(t, features['spectral_centroid'], color='#F18F01', linewidth=2)
        plt.title('Spectral Centroid (Brightness)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.grid(True, alpha=0.3)

        # 6. Zero Crossing Rate
        plt.subplot(4, 3, 6)
        t = librosa.frames_to_time(range(len(features['zero_crossing_rate'])), sr=self.sr)
        plt.plot(t, features['zero_crossing_rate'], color='#C73E1D', linewidth=2)
        plt.title('Zero Crossing Rate (Noisiness)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Rate')
        plt.grid(True, alpha=0.3)

        # 7. RMS Energy
        plt.subplot(4, 3, 7)
        t = librosa.frames_to_time(range(len(features['rms'])), sr=self.sr)
        plt.plot(t, features['rms'], color='#06A77D', linewidth=2)
        plt.fill_between(t, 0, features['rms'], alpha=0.3, color='#06A77D')
        plt.title('RMS Energy (Loudness)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)

        # 8. Pitch Contour
        plt.subplot(4, 3, 8)
        pitch_times = librosa.times_like(features['pitch'], sr=self.sr)
        plt.plot(pitch_times, features['pitch'], color='#7209B7', linewidth=2)
        plt.title('Pitch Contour (Fundamental Frequency)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True, alpha=0.3)

        # 9. MFCCs
        plt.subplot(4, 3, 9)
        librosa.display.specshow(features['mfccs'], sr=self.sr, x_axis='time', cmap='coolwarm')
        plt.colorbar()
        plt.title('MFCCs (Timbre)', fontsize=12, fontweight='bold')
        plt.ylabel('MFCC Coefficients')

        # 10. Chroma Features
        plt.subplot(4, 3, 10)
        librosa.display.specshow(features['chroma'], sr=self.sr, x_axis='time', y_axis='chroma', cmap='Greens')
        plt.colorbar()
        plt.title('Chromagram (Pitch Classes)', fontsize=12, fontweight='bold')

        # 11. Spectral Rolloff
        plt.subplot(4, 3, 11)
        t = librosa.frames_to_time(range(len(features['spectral_rolloff'])), sr=self.sr)
        plt.plot(t, features['spectral_rolloff'], color='#F72585', linewidth=2)
        plt.title('Spectral Rolloff', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.grid(True, alpha=0.3)

        # 12. Emotion Detection Results
        plt.subplot(4, 3, 12)
        emotions = list(emotion_scores.keys())
        scores = list(emotion_scores.values())
        colors = ['#FF6B6B' if e == emotion else '#4ECDC4' for e in emotions]
        bars = plt.bar(emotions, scores, color=colors, edgecolor='black', linewidth=1.5)
        plt.title(f'Detected Emotion: {emotion}', fontsize=12, fontweight='bold')
        plt.ylabel('Confidence (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig, emotion, emotion_scores, features

    def get_statistics(self, features, emotion, emotion_scores):
        """Get detailed audio statistics"""
        tempo = float(features['tempo']) if isinstance(features['tempo'], np.ndarray) else features['tempo']
        
        stats = {
            "duration": self.duration,
            "sample_rate": self.sr,
            "samples": len(self.y),
            "detected_emotion": emotion,
            "emotion_scores": emotion_scores,
            "mean_pitch": np.mean(features['pitch'][~np.isnan(features['pitch'])]),
            "mean_energy": np.mean(features['rms']),
            "tempo": tempo,
            "mean_spectral_centroid": np.mean(features['spectral_centroid']),
            "mean_zcr": np.mean(features['zero_crossing_rate']),
            "mean_spectral_bandwidth": np.mean(features['spectral_bandwidth']),
            "mean_spectral_rolloff": np.mean(features['spectral_rolloff'])
        }
        return stats

# Streamlit App Configuration
API_BASE = "https://testrakshit-nlpproject.hf.space"

st.set_page_config(
    page_title="EchoCheck - Social Media Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Custom CSS for professional styling
def load_css():
    if st.session_state.theme == 'dark':
        primary_color = "#6366f1"
        bg_color = "#0f172a"
        secondary_bg = "#1e293b"
        text_color = "#e2e8f0"
        border_color = "#334155"
        card_bg = "#1e293b"
        gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    else:
        primary_color = "#6366f1"
        bg_color = "#ffffff"
        secondary_bg = "#f8fafc"
        text_color = "#1e293b"
        border_color = "#e2e8f0"
        card_bg = "#ffffff"
        gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    
    st.markdown(f"""
    <style>
        /* Main theme styling */
        .stApp {{
            background-color: {bg_color};
        }}
        
        /* Header styling */
        .main-header {{
            background: {gradient};
            padding: 2.5rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            text-align: center;
        }}
        
        .main-header h1 {{
            color: white;
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .main-header p {{
            color: rgba(255,255,255,0.95);
            font-size: 1.2rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }}
        
        /* Card styling */
        .analysis-card {{
            background: {card_bg};
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid {border_color};
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        /* Input section styling */
        .input-section {{
            background: {secondary_bg};
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            border: 1px solid {border_color};
        }}
        
        /* Custom button */
        .stButton>button {{
            background: {gradient};
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            width: 100%;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        /* Metric cards */
        .metric-card {{
            background: {card_bg};
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid {border_color};
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {primary_color};
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: {text_color};
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* Section headers */
        .section-header {{
            color: {text_color};
            font-size: 1.5rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {primary_color};
            display: inline-block;
        }}
        
        /* Risk badge */
        .risk-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            margin: 0.25rem;
        }}
        
        .risk-low {{
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }}
        
        .risk-medium {{
            background: rgba(251, 191, 36, 0.2);
            color: #fbbf24;
        }}
        
        .risk-high {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}
        
        /* Upload area */
        .uploadedFile {{
            border: 2px dashed {border_color};
            border-radius: 12px;
            padding: 1rem;
        }}
        
        /* Emotion badges */
        .emotion-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1.25rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 1.1rem;
            margin: 0.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* News cards */
        .news-card {{
            background: {card_bg};
            border-left: 4px solid {primary_color};
            border-radius: 8px;
            padding: 1.25rem;
            margin: 1rem 0;
        }}
        
        /* Theme toggle */
        .theme-toggle {{
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 999;
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {secondary_bg};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {primary_color};
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #4f46e5;
        }}
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
        }}
        
        /* Info boxes */
        .info-box {{
            background: rgba(99, 102, 241, 0.1);
            border-left: 4px solid {primary_color};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        /* Success/Warning/Error styling */
        .stSuccess, .stWarning, .stError, .stInfo {{
            border-radius: 12px;
            padding: 1rem;
        }}
    </style>
    """, unsafe_allow_html=True)

load_css()

# Sidebar for theme toggle and info
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    theme_button = st.button("🌓 Toggle Theme", use_container_width=True, key="theme_toggle")
    if theme_button:
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **EchoCheck** analyzes your social media content before you post to help you:
    
    - 🎯 Understand audience perception
    - ⚠️ Identify potential risks
    - 💡 Get content suggestions
    - 📊 Analyze voice emotions
    - 🖼️ Extract media insights
    """)
    
    st.markdown("---")
    st.markdown("### 🚀 Features")
    st.markdown("""
    - **Caption Analysis**: Text sentiment & keywords
    - **Image Analysis**: Visual content extraction
    - **Video Analysis**: Frame & audio processing
    - **Voice Analysis**: Emotion detection from audio
    - **News Correlation**: Related current events
    - **Risk Assessment**: Content safety scoring
    """)
    
    st.markdown("---")
    st.markdown("### 💻 Tech Stack")
    st.markdown("""
    - Streamlit
    - Librosa (Audio Analysis)
    - Computer Vision APIs
    - NLP Models
    """)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎯 EchoCheck</h1>
    <p>Hear The Echoes Before You Post - AI-Powered Social Media Analysis</p>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### ✏️ Your Content")
    caption = st.text_area(
        "Caption",
        placeholder="What's on your mind? Type your caption here...",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 📎 Attachments")
    uploaded_file = st.file_uploader(
        "Upload Media",
        type=["jpg", "jpeg", "png", "mp4", "mov"],
        label_visibility="collapsed"
    )
    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        label_visibility="collapsed"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Analyze button
if st.button("🔍 Analyze Content", use_container_width=True):
    if not caption and not uploaded_file and not audio_file:
        st.warning("⚠️ Please enter a caption or upload media/audio to analyze.")
    else:
        with st.spinner("🔄 Analyzing your content... This may take a moment."):
            contexts = {"caption_context": "", "image_context": "", "video_context": "", "audio_context": ""}

            # --- Voice Emotion Analysis ---
            if audio_file:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<h2 class="section-header">🎤 Voice Emotion Analysis</h2>', unsafe_allow_html=True)
                
                file_extension = os.path.splitext(audio_file.name)[1]
                if not file_extension:
                    file_extension = '.wav'
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    with st.spinner("Processing audio file..."):
                        detector = VoiceEmotionDetector(tmp_path)
                        fig, emotion, emotion_scores, features = detector.visualize_all()
                        stats = detector.get_statistics(features, emotion, emotion_scores)
                    
                    # Emotion result with badge
                    emotion_colors = {
                        'Happy': '#22c55e',
                        'Sad': '#3b82f6',
                        'Angry': '#ef4444',
                        'Neutral': '#64748b',
                        'Fearful': '#f59e0b',
                        'Surprised': '#a855f7'
                    }
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <div class="emotion-badge" style="background: {emotion_colors.get(emotion, '#6366f1')}20; color: {emotion_colors.get(emotion, '#6366f1')};">
                            <span style="font-size: 2rem; margin-right: 0.5rem;">😊</span>
                            <span>{emotion}</span>
                        </div>
                        <div style="margin-top: 1rem; font-size: 1.2rem; opacity: 0.8;">
                            Confidence: <strong>{emotion_scores[emotion]:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Audio statistics in metric cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    metrics = [
                        ("⏱️", "Duration", f"{stats['duration']:.2f}s", col1),
                        ("🎵", "Tempo", f"{stats['tempo']:.0f} BPM", col2),
                        ("🎼", "Pitch", f"{stats['mean_pitch']:.0f} Hz", col3),
                        ("📊", "Energy", f"{stats['mean_energy']:.4f}", col4)
                    ]
                    
                    for icon, label, value, col in metrics:
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="font-size: 2rem;">{icon}</div>
                                <div class="metric-value">{value}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Emotion confidence scores
                    st.markdown("#### 📊 Detailed Emotion Analysis")
                    cols = st.columns(6)
                    for i, (emo, score) in enumerate(emotion_scores.items()):
                        with cols[i]:
                            st.metric(emo, f"{score:.1f}%", delta=None)
                    
                    # Visualization
                    st.markdown("#### 🎨 Audio Visualizations")
                    st.pyplot(fig)
                    
                    contexts["audio_context"] = f"Voice emotion: {emotion} (confidence: {emotion_scores[emotion]:.1f}%). Audio features: pitch {stats['mean_pitch']:.1f}Hz, energy {stats['mean_energy']:.4f}, tempo {stats['tempo']:.1f}BPM"
                    
                except Exception as e:
                    st.error(f"❌ Error in voice emotion analysis: {str(e)}")
                    with st.expander("🔍 View Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
                        st.info("""
                        **Troubleshooting Tips:**
                        - Ensure the audio file is not corrupted
                        - Supported formats: WAV, MP3, M4A, FLAC, OGG
                        - Try converting your audio to WAV format
                        - File size should be under 200MB
                        - Check if the audio has actual content (not silent)
                        """)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                st.markdown('</div>', unsafe_allow_html=True)

            # Capture caption
            if caption:
                contexts["caption_context"] = caption

            # --- Media Analysis ---
            if uploaded_file:
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                file_type = uploaded_file.type

                if "image" in file_type:
                    st.markdown('<h2 class="section-header">🖼️ Image Analysis</h2>', unsafe_allow_html=True)
                    files = {"image": uploaded_file.getvalue()}
                    resp = requests.post(f"{API_BASE}/generate_image_caption", files=files)
                    if resp.status_code == 200:
                        img_cap = resp.json()["caption"]
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>📝 Extracted Caption:</strong><br>
                            {img_cap}
                        </div>
                        """, unsafe_allow_html=True)
                        contexts["image_context"] = img_cap

                elif "video" in file_type:
                    st.markdown('<h2 class="section-header">🎞️ Video Analysis</h2>', unsafe_allow_html=True)
                    files = {"video": uploaded_file.getvalue()}

                    resp1 = requests.post(f"{API_BASE}/generate_video_captions", files=files)
                    if resp1.status_code == 200:
                        st.markdown("**🎬 Video Frame Captions**")
                        for k, v in resp1.json()["captions"].items():
                            st.markdown(f"- **{k}**: {v}")
                        contexts["video_context"] = str(resp1.json())

                    resp2 = requests.post(f"{API_BASE}/transcribe_video_audio", files=files, data={"model_size": "base"})
                    if resp2.status_code == 200:
                        st.markdown("**🎙️ Audio Transcription**")
                        st.markdown(f"""
                        <div class="info-box">
                            {resp2.json()["transcription"]}
                        </div>
                        """, unsafe_allow_html=True)
                        contexts["audio_context"] = resp2.json()["transcription"]
                
                st.markdown('</div>', unsafe_allow_html=True)

            # --- Complete Analysis ---
            if caption or uploaded_file:
                resp = requests.post(f"{API_BASE}/generate_keywords", data=contexts)
                keywords = resp.json().get("keywords_response", "")

                payload = {"keywords_response": keywords}
                payload.update(contexts)
                resp = requests.post(f"{API_BASE}/complete_analysis", data=payload)

                if resp.status_code == 200:
                    analysis = resp.json()

                    # News Articles
                    if analysis["news_articles"]:
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown('<h2 class="section-header">📰 Related News & Trends</h2>', unsafe_allow_html=True)
                        for a in analysis["news_articles"]:
                            st.markdown(f"""
                            <div class="news-card">
                                <h4 style="margin: 0 0 0.5rem 0;">
                                    <a href="{a['url']}" target="_blank" style="text-decoration: none; color: #6366f1;">
                                        {a['title']}
                                    </a>
                                </h4>
                                <p style="margin: 0.5rem 0; opacity: 0.8;">{a['description']}</p>
                                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.7;">
                                    <span>📰 {a['source']}</span>
                                    <span>🕐 {a['published_at']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Summary Points
                    if analysis["summary_points"]:
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        st.markdown('<h2 class="section-header">📌 Key Insights</h2>', unsafe_allow_html=True)
                        for p in analysis["summary_points"]:
                            st.markdown(f"✓ {p}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Risks
                    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-header">⚠️ Risk Assessment</h2>', unsafe_allow_html=True)
                    if analysis["risks"]:
                        for r in analysis["risks"]:
                            st.markdown(f"""
                            <div style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(239, 68, 68, 0.1); border-left: 3px solid #ef4444; border-radius: 5px;">
                                🔴 {r}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="padding: 1rem; background: rgba(34, 197, 94, 0.1); border-left: 3px solid #22c55e; border-radius: 5px;">
                            ✅ No significant risks identified. Your content appears safe to post!
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Suggestions
                    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                    st.markdown('<h2 class="section-header">💡 Smart Suggestions</h2>', unsafe_allow_html=True)
                    sugg = analysis["suggestions"]
                    
                    # Create tabs for different suggestion types
                    tab1, tab2, tab3, tab4 = st.tabs(["👥 Audience POVs", "✍️ Copy Edits", "🚩 Flags", "📊 Risk Score"])
                    
                    with tab1:
                        if sugg.get("povs"):
                            for p in sugg["povs"]:
                                st.markdown(f"""
                                <div style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(99, 102, 241, 0.1); border-left: 3px solid #6366f1; border-radius: 5px;">
                                    👤 {p}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No specific audience perspectives to highlight.")
                    
                    with tab2:
                        if sugg.get("copy_edits"):
                            for e in sugg["copy_edits"]:
                                st.markdown(f"""
                                <div style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(168, 85, 247, 0.1); border-left: 3px solid #a855f7; border-radius: 5px;">
                                    ✏️ {e}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No copy edits suggested.")
                    
                    with tab3:
                        if sugg.get("flags"):
                            for f in sugg["flags"]:
                                st.markdown(f"""
                                <div style="padding: 0.75rem; margin: 0.5rem 0; background: rgba(251, 191, 36, 0.1); border-left: 3px solid #fbbf24; border-radius: 5px;">
                                    🚩 {f}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.success("No flags raised!")
                    
                    with tab4:
                        risk_score = sugg.get('risk_score', 'N/A')
                        rationale = sugg.get('rationale', 'No rationale provided.')
                        
                        # Determine risk level
                        try:
                            score_num = float(risk_score.split('/')[0]) if '/' in str(risk_score) else 0
                            if score_num <= 3:
                                risk_class = "risk-low"
                                risk_icon = "🟢"
                                risk_text = "Low Risk"
                            elif score_num <= 6:
                                risk_class = "risk-medium"
                                risk_icon = "🟡"
                                risk_text = "Medium Risk"
                            else:
                                risk_class = "risk-high"
                                risk_icon = "🔴"
                                risk_text = "High Risk"
                        except:
                            risk_class = "risk-low"
                            risk_icon = "⚪"
                            risk_text = "Unknown"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 4rem; margin-bottom: 1rem;">{risk_icon}</div>
                            <div class="risk-badge {risk_class}" style="font-size: 1.5rem; padding: 1rem 2rem;">
                                {risk_text}: {risk_score}
                            </div>
                            <div style="margin-top: 2rem; padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 10px;">
                                <strong>📋 Rationale:</strong><br>
                                <p style="margin-top: 0.5rem;">{rationale}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error("❌ Complete analysis failed. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; opacity: 0.7;">
    <p>Made with ❤️ by EchoCheck Team</p>
    <p style="font-size: 0.9rem;">Empowering responsible social media through AI</p>
</div>
""", unsafe_allow_html=True)
