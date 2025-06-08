# Path: Yolo_PaddleOCR/main.py
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt # Import Qt for window state
from ui_main_window import UIOcrApp # Import UIOcrApp from the new file

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def main():
    app = QApplication(sys.argv)
    ex = UIOcrApp()

    # Apply a dark theme stylesheet (similar to Atom)
    app.setStyleSheet("""
        QWidget {
            background-color: #282c34; /* Atom background color */
            color: #abb2bf; /* Atom text color */
            font-family: "Segoe UI", "Liberation Sans", Arial, sans-serif; /* Common sans-serif */
        }
        QGroupBox {
            border: 1px solid #3a3f4b; /* Slightly lighter border than background */
            margin-top: 10px;
            padding-top: 10px;
            /* title color */
            color: #56b6c2; /* Cyan-like color for titles */
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left; /* Position title at top left */
            padding: 0 3px;
            color: #56b6c2; /* Ensure title text color */
        }
        QPushButton {
            background-color: #404452; /* Slightly lighter than background for buttons */
            color: #abb2bf;
            border: 1px solid #3a3f4b;
            padding: 5px;
            min-height: 20px; /* Ensure consistent button size */
        }
        QPushButton:hover {
            background-color: #454a57; /* Darker hover effect */
        }
        QPushButton:pressed {
            background-color: #3a3f4b; /* Even darker pressed effect */
        }
        QPushButton:disabled {
            background-color: #30333b; /* Disabled color */
            color: #636d83;
        }
        QLabel {
             color: #abb2bf; /* Default text color */
        }
        QTextEdit {
             background-color: #30333b; /* Darker background for text edit areas */
             color: #abb2bf;
             border: 1px solid #3a3f4b;
             padding: 5px;
        }
        QProgressBar {
            border: 1px solid #3a3f4b;
            text-align: center;
            color: #abb2bf;
            background-color: #3a3f4b;
        }
        QProgressBar::chunk {
            background-color: #61afef; /* Blue-like color for progress */
        }
        QTabWidget::pane {
            border: 1px solid #3a3f4b;
            background-color: #282c34;
        }
        QTabWidget::tab-bar {
            left: 5px; /* move to the right to make space for the left border */
        }
        QTabBar::tab {
            background: #3a3f4b; /* Darker background for inactive tabs */
            color: #abb2bf;
            border: 1px solid #3a3f4b;
            border-bottom-color: #3a3f4b; /* match the pane border */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 5px 10px;
            margin-right: 1px; /* space between tabs */
        }
        QTabBar::tab:selected {
            background: #282c34; /* Match pane background for active tab */
            border-bottom-color: #282c34; /* Hide the bottom border for the selected tab */
            color: #61afef; /* Highlight color for selected tab text */
        }
        QTabBar::tab:hover {
             background: #404452; /* Slight hover effect on tabs */
        }
    """)

    ex.show()
    # Set the window state to maximized after showing it
    ex.setWindowState(ex.windowState() | Qt.WindowState.WindowMaximized)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()