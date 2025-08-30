Smart Parking System 🚗
A comprehensive AI-powered parking management system that uses computer vision to detect and monitor parking slot occupancy in real-time using CCTV footage.

Features
Real-time Detection: YOLO-NAS based vehicle detection with DeepSort tracking

Parking Slot Management: Custom polygon-based parking slot annotation

Interactive Dashboard: Streamlit web interface for easy monitoring

Multi-format Support: Works with various video formats and custom parking layouts

Temporal Filtering: Reduces flickering and improves occupancy detection accuracy

Installation
Prerequisites
Python 3.8 or higher

Git

NVIDIA GPU (recommended for better performance)

Step-by-Step Setup
1.Clone the repository:
bash

git clone https://github.com/varuntej-21/smart-parking-system.git
cd smart-parking-system

2.Create a virtual environment (recommended):
bash

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

3.Install dependencies:
bash
pip install -r requirements.txt


Install PyTorch with CUDA support (for GPU acceleration):
bash
# Visit https://pytorch.org/get-started/locally/ for the correct command for your system
# Example for Windows with CUDA 11.7:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117



Usage
1. Preparing Your Data
A. Using Your Own Video
Place your CCTV footage in the project folder high resolution preferred (e.g., parking_video.mp4)

For best results, use a stable video with a fixed camera angle

B. Creating Parking Slot Annotations
Option 1: Using the Desktop Annotation Tool

bash
python se.py
Click "Upload Image" to load a frame from your video

Select slot type (car/bike)

Click 4 points to define each parking slot

Save as JSON file when done

Option 2: Using the Web Annotation Tool

Run the main application: streamlit run app.py

Upload a video file

Click "Open Frame Extractor" to extract a reference frame

Click "Open Annotation Tool" to create parking slots

Save your annotations as a JSON file

2. Running the Detection System
Command Line Interface:
bash
python main.py --video path/to/your/video.mp4 --json path/to/your/parking_slots.json
Web Interface (Recommended):
bash
streamlit run app.py
Then:

Upload your video file through the sidebar

Upload your parking slots JSON file

Click "Start Parking Detection"

View real-time results in the dashboard

3. Using the Frame Extractor
To get reference frames for annotation:

bash
python img.py your_video.mp4 -i  # Interactive mode
or

bash
python img.py your_video.mp4 -t 30.5  # Extract frame at 30.5 seconds
File Structure
text
smart-parking-system/

├── app.py               
├── main.py               
├── parking.py            
├── se.py                
├── img.py               
├── requirements.txt     
├── parking_slots.json   
└── README.md            
Configuration
Video Settings
Edit main.py to adjust:

Confidence threshold (default: 0.20)

Frame skipping (default: 24x speedup)

Vehicle classes to detect (default: cars and motorcycles)

Parking Slot Settings
In parking.py, you can modify:

Slot line thickness

Colors for occupied/vacant slots

Temporal filtering buffer size

Troubleshooting
Common Issues
"Module not found" errors

Ensure you've installed all requirements: pip install -r requirements.txt

CUDA out of memory

Reduce input video resolution

Use a smaller YOLO model (yolo_nas_s instead of yolo_nas_l)

Poor detection accuracy

Adjust confidence threshold in main.py

Ensure proper lighting in your video footage

Verify parking slot annotations are accurate

Streamlit deployment issues

Check that all paths are relative, not absolute

Ensure all required files are in the repository

Performance Tips
Use GPU acceleration for better performance

For longer videos, increase frame skip value

Reduce video resolution if processing is too slow

Close other applications to free up system resources

Deployment
Local Deployment
Follow installation steps above

Run streamlit run app.py

Open http://localhost:8501 in your browser

Cloud Deployment (Streamlit Sharing)
Push your code to GitHub

Sign up at share.streamlit.io

Connect your GitHub repository

Deploy with the main file path as app.py

Docker Deployment
Build the image: docker build -t smart-parking .

Run the container: docker run -p 8501:8501 smart-parking

Access at http://localhost:8501

Contributing
Fork the repository

Create a feature branch: git checkout -b feature-name

Commit changes: git commit -am 'Add new feature'

Push to branch: git push origin feature-name

Submit a pull request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Support
For questions and support:

Check the troubleshooting section above

Open an issue on GitHub

Contact the development team

Acknowledgments
YOLO-NAS and DeepSort for object detection and tracking

Streamlit for the web interface framework

OpenCV for computer vision capabilities

IIT Tirupati for the hackathon challenge
