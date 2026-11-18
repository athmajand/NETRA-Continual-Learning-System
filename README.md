👁️ NETRA: Continual Learning AI Threat Detection Framework

NETRA (Network for Extracting Threats & Real-time Anomalies) is an adaptive, multi-task dual-stream surveillance system designed to solve "closed-world" AI limitations and catastrophic forgetting in real-time perimeter security. Built for a Master's Thesis in Artificial Intelligence, it fuses spatial environmental context with human kinematics to classify 30 complex threat anomalies.

🔗 Model Weights & Dataset: The complete 10,750-sample dataset and pre-trained EWC model weights are hosted on my Hugging Face Collection here: [Insert Your Hugging Face Collection Link]

🚀 Core Engineering Capabilities

Dual-Stream Spatial-Kinematic Fusion: Processes live video through parallel neural pathways. It extracts 2048-dim spatial features (ResNet50) for environmental context and 132-dim kinematic features (MediaPipe Pose) for human biomechanics, fusing them into a unified 256-dim latent vector to eliminate context-blind errors.

Continual Learning via EWC: Integrates Elastic Weight Consolidation (EWC) to learn new, zero-day threat classes post-deployment. By applying a Fisher Information Matrix penalty, NETRA assimilated new datasets with 0.00% core memory parameter drift, completely defeating catastrophic forgetting.

Aegis Fast-Track Intelligence Cache: A human-in-the-loop triage system providing zero-shot protection. Operators can manually isolate unrecognized anomalies; the system extracts the 256-dim vector and uses rapid Cosine Similarity matching (>0.70 threshold) to trigger immediate override alarms for future occurrences before formal retraining.

Privacy-Preserving Facial Lock: Engineered for ethical AI surveillance. Facial recognition (MediaPipe BlazeFace) remains completely dormant during normal operations, locking onto subjects only when the system mathematically confirms a perimeter breach with ≥80% confidence.

Crash-Proof Multi-Threaded GUI: Features a hardware-accelerated CustomTkinter "Command Center" dashboard. Heavy deep learning inference and DVR video scrubbing are strictly decoupled into isolated background threads to guarantee a flawless, freeze-free operator experience.

Multi-Task Decision Matrix: Utilizes a two-headed neural architecture: a high-speed "Watchdog" for binary triage (Threat vs. Normal) and a fine-grained "Analyst" head for exact 30-class categorization.

🧠 System Architecture

The pipeline processes data in real-time through standard IP/CCTV cameras or local video files:

Temporal Windowing: Continuously buffers 150 consecutive frames (representing exactly 5 seconds of activity at 30 FPS).

Spatial Processing: Resizes frames to 224x224, applies ImageNet normalization, and passes them through a frozen ResNet50 backbone.

Kinematic Processing: Tracks 33 critical human body landmarks to map trajectory and motion physics.

Classification: Embeds both data streams into a 256-dim vector, routing it through the Multi-Task Decision Matrix.

EWC Update Loop: When retraining on zero-day threats, it calculates the Fisher Information Matrix from baseline data and applies quadratic penalties to protect core knowledge.

📊 Dataset & Threat Vocabulary

The model was trained on a custom and curated dataset of 10,750 sequences (8,592 train / 2,158 test) built from UCF-Crime, UCSD Pedestrian, ShanghaiTech, and web scraping.

Currently Supported Classes (30):

Violent/Aggressive: Abuse, Arrest, Assault, Fighting, Slap, Shooting, Stab, Snatching, Weaponized, Weapon_brandishing

Theft/Intrusion: Burglary, Robbery, Shoplifting, Stealing, atm_theft, Vehicle_Intrusion, Package_Theft (Added via EWC)

Public Disturbance/Destruction: Arson, Explosion, Vandalism, Crowd_Anomaly, Stone_Pelting

Suspicious/Accidental: Fall, FenceClimbing, Kidnapping, Peeping, RoadAccidents, SuspiciousBehavior, medical_emergency, intoxicated_behaviour

Baseline: Normal

💻 Installation & Setup

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/NETRA-Continual-Learning-System.git
cd NETRA-Continual-Learning-System


2. Create a virtual environment & install dependencies:

conda create -n netra_env python=3.9
conda activate netra_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python mediapipe==0.10.21 customtkinter pillow protobuf==4.25.3


(Note: MediaPipe is pinned to 0.10.21 and Protobuf to 4.25.3 to maintain compatibility between TensorFlow components and the legacy MediaPipe Pose API).

3. Download Model Weights:
Download netra_mk2_30classes.pth and fast_track_memory_db.pt from the Hugging Face link and place them in D:\Threat_Detection_Workspace\03_Base_Model\ (or update the config paths in ooda_command_center.py).

4. Launch the Command Center:

python ooda_command_center.py


🖥️ Operational Guide (Command Center)

Start Webcam / Upload MP4: Load live streams or historical CCTV footage.

Network Feed: Input HTTP/RTSP streams from external IP cameras.

Isolate Anomaly: When a novel threat occurs, pause the video and click "Isolate Anomaly". Enter a name (e.g., "Animal_Attack") to save its 256-dim feature vector to the Aegis Cache for zero-shot real-time detection.

Neural Overseer: Access the administrative terminal to view EWC metrics, validate Fisher Matrix drift, and manage the Aegis database.

📈 Results & Performance

Baseline Accuracy (29 Classes): 89.60% (Watchdog Binary Triage)

Continual Learning Drift: 0.00% parameter drift during EWC assimilation of the 30th class ("Package_Theft").

Final System Accuracy: 84.20% Granular Multi-Class validation accuracy.

Developed for MCA Master's Thesis Project 2026.
