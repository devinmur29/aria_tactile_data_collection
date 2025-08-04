import os
import json
import cv2
import sys
import argparse
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QSlider, QPushButton, QLineEdit, 
    QHBoxLayout, QTextEdit, QComboBox, QCheckBox, QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class VideoAnnotatorWidget(QWidget):
    def __init__(self, folder_name):
        super().__init__()
        
        self.folder_name = folder_name
        self.video_path = os.path.join("ariarecordings", folder_name, f"{folder_name}.mp4")
        
        # Check if video exists
        if not os.path.exists(self.video_path):
            QMessageBox.critical(self, "Error", f"Video file not found: {self.video_path}")
            sys.exit(1)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {self.video_path}")
            sys.exit(1)
        
        # Video metadata
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Annotation data
        self.annotation_data = {
            "folder_name": folder_name,
            "video_path": self.video_path,
            "metadata": {},
            "detailed_annotations": []
        }
        
        # Phase tracking
        self.current_phase = "metadata"  # "metadata" or "detailed"
        
        # Video playback state
        self.is_paused = False
        self.current_frame = 0
        self.start = 0
        self.end = self.frame_count - 1
        
        # Detailed annotation state
        self.annotation_start_frame = None
        self.annotation_end_frame = None
        self.current_annotation_text = ""

        with open(os.path.join("ariarecordings", folder_name, f"frame_timestamps.json")) as f:
            self.frame_to_ts = json.load(f)

        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        # Main video display - adjusted for tall aspect ratio
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(600, 800)  # Taller display for 1408x2176 aspect ratio
        
        # Video slider
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(self.start)
        self.slider.setMaximum(self.end)
        self.slider.sliderMoved.connect(self.seek_video)
        
        # Playback controls
        self.play_button = QPushButton('Pause', self)
        self.play_button.clicked.connect(self.toggle_play_pause)
        
        self.back_button = QPushButton('-1 Frame', self)
        self.back_button.clicked.connect(self.back_frame)
        
        self.forward_button = QPushButton('+1 Frame', self)
        self.forward_button.clicked.connect(self.forward_frame)
        
        # Phase indicator
        self.phase_label = QLabel(f"Phase: {self.current_phase.title()}", self)
        self.phase_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Metadata section (Phase 1)
        self.metadata_group = QGroupBox("Video Metadata", self)
        self.metadata_layout = QFormLayout()
        
        self.scenario_input = QLineEdit(self)
        self.scenario_input.setPlaceholderText("e.g., Cooking, Assembly, Maintenance")
        self.metadata_layout.addRow("Scenario (Activity):", self.scenario_input)
        
        self.kit_id_input = QLineEdit(self)
        self.kit_id_input.setPlaceholderText("e.g., L01, S02")
        self.metadata_layout.addRow("Kit ID:", self.kit_id_input)
        
        self.pii_checkbox = QCheckBox("PII Captured", self)
        self.metadata_layout.addRow("PII Status:", self.pii_checkbox)
        
        self.metadata_group.setLayout(self.metadata_layout)
        
        # Detailed annotation section (Phase 2)
        self.detailed_group = QGroupBox("Detailed Annotations", self)
        self.detailed_layout = QVBoxLayout()
        
        self.annotation_text = QTextEdit(self)
        self.annotation_text.setPlaceholderText("Enter detailed annotation for the selected time range...")
        self.annotation_text.setMaximumHeight(100)
        
        self.start_frame_input = QLineEdit(self)
        self.start_frame_input.setPlaceholderText("Start frame")
        
        self.end_frame_input = QLineEdit(self)
        self.end_frame_input.setPlaceholderText("End frame")
        
        self.add_annotation_button = QPushButton("Add Annotation", self)
        self.add_annotation_button.clicked.connect(self.add_detailed_annotation)
        
        self.detailed_layout.addWidget(QLabel("Annotation:"))
        self.detailed_layout.addWidget(self.annotation_text)
        self.detailed_layout.addWidget(QLabel("Frame Range:"))
        
        frame_range_layout = QHBoxLayout()
        frame_range_layout.addWidget(QLabel("Start:"))
        frame_range_layout.addWidget(self.start_frame_input)
        frame_range_layout.addWidget(QLabel("End:"))
        frame_range_layout.addWidget(self.end_frame_input)
        self.detailed_layout.addLayout(frame_range_layout)
        
        self.detailed_layout.addWidget(self.add_annotation_button)
        
        self.detailed_group.setLayout(self.detailed_layout)
        
        # Phase control buttons
        self.next_phase_button = QPushButton("Complete Metadata & Start Detailed Annotations", self)
        self.next_phase_button.clicked.connect(self.next_phase)
        
        self.save_button = QPushButton("Save Annotations", self)
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.hide()
        
        # Left side - Video and controls
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.slider)
        video_layout.addWidget(self.phase_label)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.forward_button)
        control_layout.addWidget(self.back_button)
        video_layout.addLayout(control_layout)
        
        # Right side - Annotation interface
        annotation_layout = QVBoxLayout()
        annotation_layout.addWidget(self.metadata_group)
        annotation_layout.addWidget(self.detailed_group)
        annotation_layout.addWidget(self.next_phase_button)
        annotation_layout.addWidget(self.save_button)
        annotation_layout.addStretch()  # Add stretch to push everything to the top
        
        # Main horizontal layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(annotation_layout)
        
        self.setLayout(main_layout)
        
        # Initially hide detailed annotation section
        self.detailed_group.hide()
        
    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.fps))
        
    def update_frame(self):
        if not self.is_paused and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and (self.current_frame < self.end):
                self.current_frame += 1
                self.slider.setValue(self.current_frame)
                
                # Resize and display frame - adjusted for tall aspect ratio
                frame = cv2.resize(frame, (600, 800), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cv2.putText(frame,  
                    f"frame: {self.current_frame}",  
                    (50, 50),  
                    cv2.FONT_HERSHEY_SIMPLEX, 1,  
                    (0, 0, 0),  
                    4,  
                    cv2.LINE_4)
                
                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(image))
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
                self.current_frame = self.start
    
    def back_frame(self):
        if self.current_frame > self.start:
            self.seek_video(self.current_frame - 1)

    def forward_frame(self):
        if self.current_frame < self.end:
            self.seek_video(self.current_frame + 1)

    def toggle_play_pause(self):
        if self.is_paused:
            self.play_button.setText('Pause')
            self.timer.start(int(1000 / self.fps))
        else:
            self.play_button.setText('Play')
            self.timer.stop()
        self.is_paused = not self.is_paused

    def seek_video(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        self.current_frame = position

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (600, 800), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.putText(frame,  
                f"frame: {self.current_frame}",  
                (50, 50),  
                cv2.FONT_HERSHEY_SIMPLEX, 1,  
                (0, 0, 0),  
                4,  
                cv2.LINE_4)
            
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(image))
    
    def next_phase(self):
        if self.current_phase == "metadata":
            # Validate metadata
            if not self.scenario_input.text().strip():
                QMessageBox.warning(self, "Warning", "Please enter a scenario/activity.")
                return
            if not self.kit_id_input.text().strip():
                QMessageBox.warning(self, "Warning", "Please enter a kit ID.")
                return
            
            # Save metadata
            self.annotation_data["metadata"] = {
                "scenario": self.scenario_input.text().strip(),
                "kit_id": self.kit_id_input.text().strip(),
                "pii_captured": self.pii_checkbox.isChecked(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Switch to detailed annotation phase
            self.current_phase = "detailed"
            self.phase_label.setText(f"Phase: {self.current_phase.title()}")
            self.metadata_group.hide()
            self.detailed_group.show()
            self.next_phase_button.setText("Save Annotations")
            self.next_phase_button.clicked.disconnect()
            self.next_phase_button.clicked.connect(self.save_annotations)
            
            # Reset video to beginning for detailed annotation
            self.seek_video(0)
            
        else:
            self.save_annotations()
    
    def add_detailed_annotation(self):
        try:
            start_frame = int(self.start_frame_input.text())
            end_frame = int(self.end_frame_input.text())
            annotation_text = self.annotation_text.toPlainText().strip()
            
            if not annotation_text:
                QMessageBox.warning(self, "Warning", "Please enter annotation text.")
                return
            
            if start_frame >= end_frame:
                QMessageBox.warning(self, "Warning", "Start frame must be less than end frame.")
                return
            
            if start_frame < 0 or end_frame >= self.frame_count:
                QMessageBox.warning(self, "Warning", "Frame numbers out of range.")
                return
            
            # Add annotation
            annotation = {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_timestamp": self.frame_to_ts[str(start_frame)],
                "end_timestamp": self.frame_to_ts[str(end_frame)],
                "annotation": annotation_text,
                "timestamp": datetime.now().isoformat()
            }
            
            self.annotation_data["detailed_annotations"].append(annotation)
            
            # Clear inputs
            self.start_frame_input.clear()
            self.end_frame_input.clear()
            self.annotation_text.clear()
            
            QMessageBox.information(self, "Success", f"Added annotation for frames {start_frame}-{end_frame}")
            
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid frame numbers.")
    
    def save_annotations(self):
        output_file = os.path.join("ariarecordings", self.folder_name, f"{self.folder_name}_annotations.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(self.annotation_data, f, indent=2)
            
            QMessageBox.information(self, "Success", f"Annotations saved to {output_file}")
            self.close()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {str(e)}")
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='Video Annotation Tool')
    parser.add_argument('--foldername', help='Name of the folder containing the video (folder_name/folder_name.mp4)')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    annotator = VideoAnnotatorWidget(args.foldername)
    annotator.setWindowTitle(f'Video Annotator - {args.foldername}')
    annotator.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
