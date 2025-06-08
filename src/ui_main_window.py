import sys
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout,
                             QLabel, QFileDialog, QMessageBox, QSizePolicy, QGroupBox, QTextEdit,
                             QProgressBar, QTabWidget, QLineEdit, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QWheelEvent, QPen
from PyQt6.QtCore import Qt, QRectF, QPoint, pyqtSignal, QPointF
import cv2
import json
from datetime import datetime

# Import modular components
from ui_widgets import ZoomableLabel, draw_annotations
from analysis_core import AnalysisCore
from data_manager import DataManager
from gemini_handler import GeminiHandler


class UIOcrApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UI Element and Text Detector")
        self.setGeometry(100, 100, 1200, 800) # Increased window size

        # Initialize paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.output_dir = os.path.join(self.base_dir, 'output')
        self.config_dir = os.path.join(self.base_dir, 'config')

        # Ensure directories exist
        for directory in [self.models_dir, self.output_dir, self.config_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Initialize last directory
        app_data_dir = os.path.join(os.path.expanduser("~"), '.yolo_paddle_ocr')
        if not os.path.exists(app_data_dir):
            try:
                os.makedirs(app_data_dir)
            except Exception as e:
                print(f"Error creating config directory: {e}")
                app_data_dir = os.path.expanduser("~")
        
        self.config_file = os.path.join(app_data_dir, 'app_config.json')
        self.last_directory = self._load_last_directory()

        self.layout = QVBoxLayout()

        # --- Tab Widget for Outputs ---
        self.output_tabs = QTabWidget()
        self.layout.addWidget(self.output_tabs, 1) # Give tabs area more space

        # Tab 1: All Annotations + Info/JSON
        self.all_tab = QWidget()
        self.all_tab_layout = QHBoxLayout()
        self.all_tab.setLayout(self.all_tab_layout)
        self.output_tabs.addTab(self.all_tab, "All Annotations + Info")

        # Image display for All Annotations
        self.all_image_label = ZoomableLabel() # Use custom label for all annotations
        self.all_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.all_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.all_tab_layout.addWidget(self.all_image_label, 2) # Takes 2/3 of space

        # Info and Controls Panel group - now within the All tab's layout
        info_controls_group = QGroupBox("Info and JSON Output") # Renamed group
        info_controls_layout = QVBoxLayout()
        info_controls_group.setLayout(info_controls_layout)
        self.all_tab_layout.addWidget(info_controls_group, 1) # Takes 1/3 of space

        # Add Load and Process buttons to the Info/Controls layout
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        info_controls_layout.addWidget(self.load_button)

        self.process_button = QPushButton("Run Analysis") # Renamed button text
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        info_controls_layout.addWidget(self.process_button)
        info_controls_layout.addStretch(1) # Add stretch to push buttons to the top

        # Add Chat Interface
        chat_group = QGroupBox("Chat with AI Assistant")
        chat_layout = QVBoxLayout()
        chat_group.setLayout(chat_layout)
        # Make chat_group expand vertically
        chat_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        info_controls_layout.addWidget(chat_group, 2)  # Give chat more space

        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Ask questions about the detected elements...")
        self.chat_display.setMinimumHeight(200)
        chat_layout.addWidget(self.chat_display, 3)

        # Chat input area
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your question here...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        chat_input_layout.addWidget(self.chat_input)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_chat_message)
        chat_input_layout.addWidget(self.send_button)
        chat_layout.addLayout(chat_input_layout)

        # Gemini API Key input area
        api_key_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Gemini API key...")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.api_key_input)
        self.set_key_button = QPushButton("Set Key")
        self.set_key_button.clicked.connect(self.set_gemini_api_key)
        api_key_layout.addWidget(self.set_key_button)
        chat_layout.addLayout(api_key_layout)

        # --- Hide Hover Info Display ---
        # hover_info_group = QGroupBox("Hovered Element Info")
        # hover_info_layout = QVBoxLayout()
        # hover_info_group.setLayout(hover_info_layout)
        # info_controls_layout.addWidget(hover_info_group)
        # self.hover_info_text = QTextEdit() # Use QTextEdit for potentially long info
        # self.hover_info_text.setReadOnly(True)
        # self.hover_info_text.setPlaceholderText("Hover over an element or text in the image to see its details here.")
        # hover_info_layout.addWidget(self.hover_info_text)

        # Raw JSON Output Display
        json_output_group = QGroupBox("Raw Analysis JSON Output")
        json_output_layout = QVBoxLayout()
        json_output_group.setLayout(json_output_layout)
        info_controls_layout.addWidget(json_output_group)

        # Add download button container for JSON
        json_button_container = QHBoxLayout()
        self.json_download_button = QPushButton("Download JSON")
        self.json_download_button.setMaximumWidth(100)  # Make button smaller
        self.json_download_button.clicked.connect(self.download_json_output)
        json_button_container.addWidget(self.json_download_button)
        json_button_container.addStretch()  # Push button to the left
        json_output_layout.addLayout(json_button_container)

        self.json_output_text = QTextEdit() # Keep the raw JSON display
        self.json_output_text.setReadOnly(True)
        json_output_layout.addWidget(self.json_output_text)


        # Tab 2: PaddleOCR Annotations
        self.ocr_tab = QWidget()
        self.ocr_tab_layout = QVBoxLayout()
        self.ocr_tab.setLayout(self.ocr_tab_layout)
        # self.output_tabs.addTab(self.ocr_tab, "PaddleOCR Annotations")  # Hidden tab

        self.ocr_image_label = ZoomableLabel() # Use custom label for OCR annotations
        self.ocr_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ocr_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ocr_tab_layout.addWidget(self.ocr_image_label)

        # Tab 3: YOLO Annotations
        self.yolo_tab = QWidget()
        self.yolo_tab_layout = QVBoxLayout()
        self.yolo_tab.setLayout(self.yolo_tab_layout)
        # self.output_tabs.addTab(self.yolo_tab, "YOLO Annotations")  # Hidden tab

        self.yolo_image_label = ZoomableLabel() # Use custom label for YOLO annotations
        self.yolo_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.yolo_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.yolo_tab_layout.addWidget(self.yolo_image_label)

        # Tab 4: Combined Annotations (Same as All Tab image, maybe useful to have separate)
        self.combined_tab = QWidget()
        self.combined_tab_layout = QVBoxLayout()
        self.combined_tab.setLayout(self.combined_tab_layout)
        self.output_tabs.addTab(self.combined_tab, "Combined Annotated Image")

        # Add download button container
        download_container = QHBoxLayout()
        self.download_button = QPushButton("Download Annotated Image")
        self.download_button.clicked.connect(self.download_annotated_image)
        download_container.addWidget(self.download_button)
        download_container.addStretch()  # Push button to the left
        self.combined_tab_layout.addLayout(download_container)

        self.combined_image_label = ZoomableLabel() # Use custom label for combined annotations
        self.combined_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.combined_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.combined_tab_layout.addWidget(self.combined_image_label)


        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide() # Hide initially
        self.layout.addWidget(self.progress_bar) # Add to the main layout

        self.setLayout(self.layout)

        self.original_image_cv = None # Store original image as OpenCV format
        self.original_image_path = None
        self.analysis_data = None # Store the processed JSON output data

        # Store raw YOLO and OCR results for drawing
        self._yolo_results = []
        self._ocr_results = []

        # Update model path
        yolo_model_path = os.path.join(self.models_dir, 'yolov8m_for_ocr', 'weights', 'best.pt')
        ocr_params = {
            'lang': 'en',
            'use_textline_orientation': False,
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
        }
        self.analysis_core = AnalysisCore(yolo_model_path, ocr_params)
        self.data_manager = DataManager(output_dir=self.output_dir)

        # Initialize Gemini handler with config path
        gemini_config_path = os.path.join(self.config_dir, 'config.json')
        self.gemini_handler = GeminiHandler(config_path=gemini_config_path)

        # Connect the custom hover signal from each image label to a slot in this window
        self.all_image_label.image_hovered_coords.connect(self.handle_image_hover)
        self.ocr_image_label.image_hovered_coords.connect(self.handle_image_hover)
        self.yolo_image_label.image_hovered_coords.connect(self.handle_image_hover)
        self.combined_image_label.image_hovered_coords.connect(self.handle_image_hover)

        # Keep track of which tab is currently active to inform the hover handler
        self.output_tabs.currentChanged.connect(self.handle_tab_changed)
        self._active_tab_index = 0 # Default to the first tab

        # Check if models loaded successfully
        if not self.analysis_core.yolo_model or not self.analysis_core.ocr_model:
             QMessageBox.critical(self, "Model Loading Error", "One or both AI models failed to load during initialization. Processing disabled.")
             self.process_button.setEnabled(False)
        else:
             self.process_button.setEnabled(True)


    def handle_tab_changed(self, index):
        """Updates the active tab index when the user switches tabs."""
        self._active_tab_index = index
        # When the tab changes, clear the hover info and highlight as the context changes
        # Clear highlight on ALL labels
        self.all_image_label.set_highlight(None)
        self.ocr_image_label.set_highlight(None)
        self.yolo_image_label.set_highlight(None)
        self.combined_image_label.set_highlight(None)
        # self.hover_info_text.clear() # removed
        # self.hover_info_text.setPlaceholderText("Hover over an element or text in the image to see its details here.") # removed


    def _load_last_directory(self):
        """Load the last used directory from config file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    last_dir = config.get('last_directory')
                    if last_dir and os.path.exists(last_dir):
                        return last_dir
        except Exception as e:
            print(f"Error loading last directory: {e}")
            # If there's any error, try to remove the corrupted config file
            try:
                if os.path.exists(self.config_file):
                    os.remove(self.config_file)
            except:
                pass
        return os.path.expanduser("~")  # Default to user's home directory

    def _save_last_directory(self):
        """Save the last used directory to config file."""
        try:
            config = {}
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                except:
                    # If file is corrupted, start with empty config
                    config = {}
            
            config['last_directory'] = self.last_directory
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving last directory: {e}")

    def load_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Image",
            self.last_directory,  # Use the last directory
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if image_path:
            # Update last directory to the directory of the selected file
            self.last_directory = os.path.dirname(image_path)
            self._save_last_directory()  # Save the new last directory
            self.original_image_path = image_path
            self.original_image_cv = cv2.imread(image_path)

            if self.original_image_cv is None:
                QMessageBox.critical(self, "Error", "Cannot load image file using OpenCV.")
                self.original_image_path = None
                # Clear all image labels and info/json
                self.all_image_label.setPixmap(QPixmap())
                self.ocr_image_label.setPixmap(QPixmap())
                self.yolo_image_label.setPixmap(QPixmap())
                self.combined_image_label.setPixmap(QPixmap())
                self.process_button.setEnabled(False)
                # self.hover_info_text.clear() # Clear hover info (removed)
                self.json_output_text.clear() # Clear previous JSON
                self.analysis_data = None # Clear previous analysis data
                self._yolo_results = []
                self._ocr_results = []
            else:
                # Convert original OpenCV image to QPixmap for displaying initially (before annotation)
                height, width, channel = self.original_image_cv.shape
                bytes_per_line = 3 * width
                q_image = QImage(self.original_image_cv.data, width, height, bytes_per_line, QImage.Format.Format_BGR888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)

                # Set the same initial pixmap for all image labels
                self.all_image_label.setPixmap(pixmap)
                self.ocr_image_label.setPixmap(pixmap)
                self.yolo_image_label.setPixmap(pixmap)
                self.combined_image_label.setPixmap(pixmap)

                # Explicitly apply scale to ensure images are drawn correctly after loading
                self.all_image_label._apply_scale()
                self.ocr_image_label._apply_scale()
                self.yolo_image_label._apply_scale()
                self.combined_image_label._apply_scale()

                # Clear any previous highlights and info/json
                self.all_image_label.set_highlight(None)
                self.ocr_image_label.set_highlight(None)
                self.yolo_image_label.set_highlight(None)
                self.combined_image_label.set_highlight(None)
                # self.hover_info_text.clear() # removed
                self.json_output_text.clear()
                self.analysis_data = None
                self._yolo_results = []
                self._ocr_results = []

                # Enable process button only if models are loaded
                if self.analysis_core.yolo_model and self.analysis_core.ocr_model:
                    self.process_button.setEnabled(True)
                else:
                     self.process_button.setEnabled(False)

                # Check if analysis for this image is in cache
                cached_data = self.data_manager.get_cached_analysis(self.original_image_path)
                if cached_data:
                    print(f"Loading analysis from cache for {self.original_image_path}")
                    self.analysis_data = cached_data['analysis_data']
                    self._yolo_results = cached_data['yolo_results']
                    self._ocr_results = cached_data['ocr_results']

                    # Display cached JSON
                    try:
                         json_data_str = json.dumps(self.analysis_data, indent=4)
                         self.json_output_text.setText(json_data_str)
                         self.json_output_text.setPlaceholderText("")
                    except Exception as json_e:
                         self.json_output_text.setText(f"Error displaying cached JSON: {json_e}")
                         print(f"Error displaying cached JSON: {json_e}")

                    # Redraw annotations on all labels from cached results
                    self.draw_and_set_annotated_images()

                else:
                    # If not in cache, ensure process button is enabled (if models loaded)
                    if self.analysis_core.yolo_model and self.analysis_core.ocr_model:
                         self.process_button.setEnabled(True)
                    else:
                         self.process_button.setEnabled(False)


    def process_image(self):
        if self.original_image_cv is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        if not self.analysis_core.yolo_model or not self.analysis_core.ocr_model:
             QMessageBox.critical(self, "Error", "AI models failed to load. Cannot process.")
             return

        # Check if analysis is already cached for this image
        cached_data = self.data_manager.get_cached_analysis(self.original_image_path)
        if cached_data:
             print(f"Analysis for {self.original_image_path} found in cache. Skipping reprocessing.")
             self.analysis_data = cached_data['analysis_data']
             self._yolo_results = cached_data['yolo_results']
             self._ocr_results = cached_data['ocr_results']

             try:
                 json_data_str = json.dumps(self.analysis_data, indent=4)
                 self.json_output_text.setText(json_data_str)
                 self.json_output_text.setPlaceholderText("")
             except Exception as json_e:
                 self.json_output_text.setText(f"Error displaying cached JSON: {json_e}")
                 print(f"Error displaying cached JSON: {json_e}")

             self.draw_and_set_annotated_images()

             self.process_button.setEnabled(True)
             self.load_button.setEnabled(True)
             self.setCursor(Qt.CursorShape.ArrowCursor)
             # self.hover_info_text.setPlaceholderText("Hover over an element or text in the image to see its details here.") # removed
             self.json_output_text.setPlaceholderText("")
             self.progress_bar.setValue(100)
             self.progress_bar.hide()
             return # Exit the function as we used cache


        self.process_button.setEnabled(False) # Disable during processing
        self.load_button.setEnabled(False)
        self.setCursor(Qt.CursorShape.WaitCursor) # Show busy cursor
        # self.hover_info_text.setPlaceholderText("Processing...") # removed
        self.json_output_text.setPlaceholderText("Processing...")
        # Clear highlights and previous results displays on processing start
        self.all_image_label.set_highlight(None)
        self.ocr_image_label.set_highlight(None)
        self.yolo_image_label.set_highlight(None)
        self.combined_image_label.set_highlight(None)
        # self.hover_info_text.clear() # removed
        self.json_output_text.clear()
        self.analysis_data = None
        self._yolo_results = []
        self._ocr_results = []


        self.progress_bar.show() # Show the progress bar
        self.progress_bar.setValue(5) # Initial progress
        QApplication.processEvents() # Process events to update UI

        try:
            # Run analysis core
            self._yolo_results, self._ocr_results = self.analysis_core.run_analysis(self.original_image_cv)

            self.progress_bar.setValue(70) # Progress after core analysis
            QApplication.processEvents()

            # Associate results and generate final JSON
            self.analysis_data = self.analysis_core.associate_results(self._yolo_results, self._ocr_results)

            self.progress_bar.setValue(80) # Progress after association
            QApplication.processEvents()

            # Display Raw JSON Output
            try:
                 json_data_str = json.dumps(self.analysis_data, indent=4)
                 self.json_output_text.setText(json_data_str)
                 self.json_output_text.setPlaceholderText("")
            except Exception as json_e:
                 self.json_output_text.setText(f"Error displaying JSON: {json_e}")
                 print(f"Error displaying JSON: {json_e}")

            # Clear chat history when new analysis is done
            self.gemini_handler.clear_history()
            self.chat_display.clear()
            self.chat_display.setPlaceholderText("Ask questions about the detected elements...")

            # Draw Annotations and Update Image Labels
            self.draw_and_set_annotated_images()
            self.progress_bar.setValue(90) # Progress after drawing
            QApplication.processEvents()

            # Cache results
            self.data_manager.cache_analysis(self.original_image_path, self.analysis_data, self._yolo_results, self._ocr_results)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.progress_bar.setValue(100) # Set to 100% on completion/error
            self.progress_bar.hide() # Hide the progress bar
            self.process_button.setEnabled(True) # Re-enable button
            self.load_button.setEnabled(True)
            self.setCursor(Qt.CursorShape.ArrowCursor) # Restore cursor
            # self.hover_info_text.setPlaceholderText("Hover over an element or text in the image to see its details here.") # removed
            self.json_output_text.setPlaceholderText("")


    def draw_and_set_annotated_images(self):
        """Draws annotations based on current results and sets pixmaps for all labels."""
        if self.original_image_cv is None:
             return

        # Draw images for each tab using the draw_annotations helper
        # Tab 0 & 3: All Annotations (No initial annotations, only on hover) and Combined (Both annotations)
        # Use the original image for the base of all drawings
        base_image = self.original_image_cv

        # Tab 0: All Annotations + Info (No annotations drawn initially)
        all_tab_base_pixmap = draw_annotations(base_image, yolo_elements=None, ocr_text_blocks=None)
        self.all_image_label.setPixmap(all_tab_base_pixmap)

        # Tab 1: PaddleOCR Annotations only
        ocr_annotated_pixmap = draw_annotations(base_image, yolo_elements=None, ocr_text_blocks=self._ocr_results)
        self.ocr_image_label.setPixmap(ocr_annotated_pixmap)

        # Tab 2: YOLO Annotations only (with associated text)
        yolo_annotated_pixmap = draw_annotations(base_image, yolo_elements=self._yolo_results, ocr_text_blocks=None, draw_yolo_associated_text=True, draw_element_type=True)
        self.yolo_image_label.setPixmap(yolo_annotated_pixmap)

        # Tab 3: Combined Annotated Image (Both YOLO and OCR annotations)
        combined_annotated_pixmap = draw_annotations(base_image, yolo_elements=self._yolo_results, ocr_text_blocks=self._ocr_results, draw_element_type=True)
        self.combined_image_label.setPixmap(combined_annotated_pixmap)


    def image_label_for_tab_index(self, index):
        """Helper to get the correct ZoomableLabel instance for a given tab index."""
        if index == 0:
            return self.all_image_label
        elif index == 1:
            return self.ocr_image_label
        elif index == 2:
            return self.yolo_image_label
        elif index == 3:
            return self.combined_image_label
        return self.all_image_label # Default to the first tab's label


    def handle_image_hover(self, position: QPoint):
        """Handles mouse hover events on the image label and highlights the hovered element based on the active tab."""
        current_image_label = self.image_label_for_tab_index(self._active_tab_index)

        if position.x() == -1 and position.y() == -1:
            current_image_label.set_highlight(None)
            current_image_label.set_info_box(visible=False)
            # self.hover_info_text.setPlaceholderText("Hover over an element or text in the image to see its details here.") # removed
            # self.hover_info_text.clear() # removed
            return

        if self.analysis_data is None or not self.analysis_data:
            current_image_label.set_highlight(None)
            current_image_label.set_info_box(visible=False)
            # self.hover_info_text.setPlaceholderText("No analysis data loaded.") # removed
            return

        # Always check the full analysis_data for hover info
        items_to_check_for_hover = self.analysis_data
        position_f = QPointF(position)
        hovered_item = None

        for item in reversed(items_to_check_for_hover):
             if 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                  try:
                      x1, y1, x2, y2 = item['bbox']
                      bbox_rect = QRectF(x1, y1, x2 - x1, y2 - y1)

                      if bbox_rect.contains(position_f):
                           item_type = item.get('type')
                           is_yolo_element = item_type != 'text'
                           is_ocr_text = item_type == 'text'

                           # Filter based on active tab for highlighting but use combined data for info
                           if self._active_tab_index == 0 or self._active_tab_index == 3: # All or Combined
                                hovered_item = item
                                break
                           elif self._active_tab_index == 1 and is_ocr_text: # OCR Tab
                                hovered_item = item
                                break
                           elif self._active_tab_index == 2 and is_yolo_element: # YOLO Tab
                                hovered_item = item
                                break
                  except (TypeError, ValueError):
                      continue

        # Clear previous highlight on ALL labels
        self.all_image_label.set_highlight(None)
        self.ocr_image_label.set_highlight(None)
        self.yolo_image_label.set_highlight(None)
        self.combined_image_label.set_highlight(None)

        # Clear info boxes on all labels
        self.all_image_label.set_info_box(visible=False)
        self.ocr_image_label.set_info_box(visible=False)
        self.yolo_image_label.set_info_box(visible=False)
        self.combined_image_label.set_info_box(visible=False)

        if hovered_item:
            # Determine highlight color based on item type
            item_type = hovered_item.get('type')
            if item_type == 'text':
                 highlight_color = QColor(0, 0, 255) # Blue for text highlight
            elif item_type != 'text': # Assuming non-text types are YOLO elements
                 highlight_color = QColor(0, 255, 0) # Green for YOLO element highlight
            else: # Fallback color
                 highlight_color = QColor(255, 0, 0) # Red

            highlight_thickness = 3
            current_image_label.set_highlight(hovered_item['bbox'], highlight_color, highlight_thickness)

            # Show info box only in All Annotations tab
            if self._active_tab_index == 0:
                current_image_label.set_info_box(hovered_item, True)

            info_text = f"Type: {hovered_item.get('type', 'N/A')}\n"
            if 'confidence' in hovered_item:
                 info_text += f"Confidence: {hovered_item['confidence']:.2f}\n"
            # Format bbox coordinates to 2 decimal places
            bbox = hovered_item.get('bbox', ['N/A', 'N/A', 'N/A', 'N/A'])
            if isinstance(bbox, list) and all(isinstance(x, (int, float)) for x in bbox):
                formatted_bbox = [f"{coord:.2f}" for coord in bbox]
                info_text += f"Bbox: [{', '.join(formatted_bbox)}]\n"
            else:
                info_text += f"Bbox: {bbox}\n"

            if hovered_item.get('type') == 'text':
                 info_text += f"Text: {hovered_item.get('text', 'N/A')}\n"
            elif 'associated_text' in hovered_item and hovered_item['associated_text']:
                 info_text += "Associated Text:\n"
                 sorted_associated_text = sorted(hovered_item['associated_text'], key=lambda x: x.get('bbox', [0])[1])
                 for assoc_text in sorted_associated_text:
                     # Format associated text bbox coordinates
                     assoc_bbox = assoc_text.get('bbox', ['N/A', 'N/A', 'N/A', 'N/A'])
                     if isinstance(assoc_bbox, list) and all(isinstance(x, (int, float)) for x in assoc_bbox):
                         formatted_assoc_bbox = [f"{coord:.2f}" for coord in assoc_bbox]
                         info_text += f"  - {assoc_text.get('text', 'N/A')} (Conf: {assoc_text.get('confidence', 0):.2f}, Bbox: [{', '.join(formatted_assoc_bbox)}])\n"
                     else:
                         info_text += f"  - {assoc_text.get('text', 'N/A')} (Conf: {assoc_text.get('confidence', 0):.2f})\n"

            # self.hover_info_text.setText(info_text) # removed
        else:
            current_image_label.set_highlight(None)
            current_image_label.set_info_box(visible=False)
            # self.hover_info_text.setPlaceholderText("Hover over an element or text in the image to see its details here.") # removed
            # self.hover_info_text.clear() # removed


    def keyPressEvent(self, event):
        """Handle key press events for zooming."""
        # Check for Ctrl + '+' (or '=') for zoom in
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and (event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal):
            current_label = self.image_label_for_tab_index(self._active_tab_index)
            if current_label:
                current_label.zoom_in()
                event.accept() # Accept the event to prevent further propagation
        # Check for Ctrl + '-' for zoom out
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Minus:
            current_label = self.image_label_for_tab_index(self._active_tab_index)
            if current_label:
                current_label.zoom_out()
                event.accept() # Accept the event
        else:
            # For other keys, call the base class implementation
            super().keyPressEvent(event)

    def send_chat_message(self):
        """Handle sending chat messages to Gemini."""
        if not self.analysis_data:
            QMessageBox.warning(self, "Warning", "Please run analysis first before asking questions.")
            return

        user_query = self.chat_input.text().strip()
        if not user_query:
            return

        # Display user message
        self.chat_display.append(f"\nYou: {user_query}")
        self.chat_input.clear()

        # Get image name from the current image path
        image_name = os.path.basename(self.original_image_path) if self.original_image_path else None
        response = self.gemini_handler.generate_response(user_query, self.analysis_data, image_name)
        
        # Display AI response
        self.chat_display.append(f"\nAI: {response}\n")
        
        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def download_annotated_image(self):
        """Handle downloading the annotated image."""
        if not self.combined_image_label._current_pixmap:
            QMessageBox.warning(self, "Warning", "No annotated image available to download.")
            return

        # Get the current image name without extension
        base_name = os.path.splitext(os.path.basename(self.original_image_path))[0] if self.original_image_path else "annotated_image"
        
        # Create default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{base_name}_annotated_{timestamp}.png"
        
        # Open file dialog for saving
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Annotated Image",
            os.path.join(self.last_directory, default_filename),
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*.*)"
        )

        if file_path:
            try:
                # Save the current pixmap
                self.combined_image_label._current_pixmap.save(file_path)
                QMessageBox.information(self, "Success", f"Image saved successfully to:\n{file_path}")
                
                # Update last directory to the save location
                self.last_directory = os.path.dirname(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def download_json_output(self):
        """Handle downloading the JSON analysis output."""
        if not self.analysis_data:
            QMessageBox.warning(self, "Warning", "No analysis data available to download.")
            return

        # Get the current image name without extension
        base_name = os.path.splitext(os.path.basename(self.original_image_path))[0] if self.original_image_path else "analysis"
        
        # Create default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{base_name}_analysis_{timestamp}.json"
        
        # Open file dialog for saving
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Save Analysis JSON",
            os.path.join(self.last_directory, default_filename),
            "JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            try:
                # Save the JSON data with proper formatting
                with open(file_path, 'w') as f:
                    json.dump(self.analysis_data, f, indent=4)
                QMessageBox.information(self, "Success", f"JSON data saved successfully to:\n{file_path}")
                
                # Update last directory to the save location
                self.last_directory = os.path.dirname(file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save JSON data: {str(e)}")

    def set_gemini_api_key(self):
        """Set and save the Gemini API key from the input box."""
        new_key = self.api_key_input.text().strip()
        if not new_key:
            QMessageBox.warning(self, "Input Error", "Please enter a valid Gemini API key.")
            return
        # Save to config.json
        config_path = os.path.join(self.config_dir, 'config.json')
        try:
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            config['gemini_api_key'] = new_key
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            QMessageBox.information(self, "Gemini API Key", "API key saved. The AI assistant will use this key from now on.")
            # Re-initialize GeminiHandler with new key
            self.gemini_handler = GeminiHandler(config_path=config_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save API key: {e}")
