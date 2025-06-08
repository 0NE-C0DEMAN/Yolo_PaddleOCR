# Path: Yolo_PaddleOCR/ui_widgets.py
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QPainter, QColor, QWheelEvent, QPen, QImage, QFont
from PyQt6.QtCore import Qt, QRectF, QPoint, pyqtSignal, QPointF
import cv2
import numpy as np


# --- ZoomableLabel Class ---
# This class is needed to handle image scaling and mouse events for accurate coordinate mapping
class ZoomableLabel(QLabel):
    # Custom signal to emit original image coordinates on mouse hover
    image_hovered_coords = pyqtSignal(QPoint) # Signal defined within the label
    # Signal to show/hide info box
    show_info_box = pyqtSignal(dict, bool)  # Emits hover info and visibility state

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True) # Enable mouse tracking
        self._original_pixmap = None # This will be the base pixmap (possibly annotated)
        self._current_pixmap = None # Scaled pixmap currently displayed
        self._initial_fit_scale = 1.0 # Scale factor when image initially fits the label
        self._current_scale = 1.0 # Keep track of the current overall scale relative to original pixmap

        self._highlight_bbox = None # Bbox to highlight [x1, y1, x2, y2] in original image coordinates
        self._highlight_color = QColor(255, 0, 0) # Default highlight color (Red)
        self._highlight_thickness = 2

        # Info box properties
        self._info_box_visible = False
        self._info_box_data = None
        self._info_box_position = None
        self._info_box_size = None

        # Panning variables
        self._is_panning = False
        self._last_mouse_pos = None
        self._pan_offset = QPointF(0, 0)
        self._total_pan_offset = QPointF(0, 0)

        # Connect to mouse move event (will be handled by parent initially)
        # self.mouseMoveEvent = self.mouse_move_event # We will connect this from the parent widget


    def setPixmap(self, pixmap: QPixmap):
        # Store the provided pixmap as the 'original' for this label (it's possibly annotated)
        self._original_pixmap = pixmap
        # Calculate the initial fit scale when the pixmap is first set
        self._initial_fit_scale = self._calculate_fit_scale()
        self._current_scale = self._initial_fit_scale # Start with the initial fit scale
        self._total_pan_offset = QPointF(0, 0) # Reset pan offset when new image is loaded
        self._apply_scale() # Apply this initial scale

    def _calculate_fit_scale(self):
         """Calculates the scale needed to fit the original pixmap into the current label size."""
         if self._original_pixmap:
             label_size = self.size()
             if label_size.width() <= 0 or label_size.height() <= 0:
                 return 1.0 # Default scale if label size is invalid

             pixmap_size = self._original_pixmap.size()
             if pixmap_size.width() <= 0 or pixmap_size.height() <= 0:
                  return 1.0 # Default scale if pixmap size is invalid

             width_scale = label_size.width() / pixmap_size.width()
             height_scale = label_size.height() / pixmap_size.height()
             return min(width_scale, height_scale)
         return 1.0 # Default scale if no original pixmap


    def _apply_scale(self):
        """Scales the original pixmap and draws it, including highlight if any."""
        if self._original_pixmap and self._current_scale > 0: # Ensure valid scale
            new_width = int(self._original_pixmap.width() * self._current_scale)
            new_height = int(self._original_pixmap.height() * self._current_scale)

            # Scale the original pixmap to the new size
            scaled_pixmap = self._original_pixmap.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

            # Create a QImage from the scaled pixmap to draw on
            image_to_draw = scaled_pixmap.toImage()

            # Create a painter for the image
            painter = QPainter(image_to_draw)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # --- Draw Highlighted Bbox (if any) ---
            if self._highlight_bbox: # Check if a highlight bbox is set
                # Scale the highlight bbox from original image coordinates to current scaled image coordinates
                x1, y1, x2, y2 = self._highlight_bbox
                scaled_x1 = int(x1 * self._current_scale)
                scaled_y1 = int(y1 * self._current_scale)
                scaled_x2 = int(x2 * self._current_scale)
                scaled_y2 = int(y2 * self._current_scale)

                # Create a semi-transparent dark overlay for the entire image
                overlay_color = QColor(0, 0, 0, 26)  # Black with 10% opacity
                
                # Draw the overlay in four rectangles around the highlighted area
                # Top rectangle
                painter.fillRect(0, 0, image_to_draw.width(), scaled_y1, overlay_color)
                # Bottom rectangle
                painter.fillRect(0, scaled_y2, image_to_draw.width(), image_to_draw.height() - scaled_y2, overlay_color)
                # Left rectangle
                painter.fillRect(0, scaled_y1, scaled_x1, scaled_y2 - scaled_y1, overlay_color)
                # Right rectangle
                painter.fillRect(scaled_x2, scaled_y1, image_to_draw.width() - scaled_x2, scaled_y2 - scaled_y1, overlay_color)

                # Draw the highlight border
                highlight_pen = QPen(QColor(self._highlight_color))
                highlight_pen.setWidth(self._highlight_thickness)
                painter.setPen(highlight_pen)
                painter.drawRect(scaled_x1, scaled_y1, scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)

            painter.end()

            # Set the QImage (with or without highlight) back as the pixmap
            self._current_pixmap = QPixmap.fromImage(image_to_draw)

            self.update() # Request a repaint after updating the pixmap

        else:
            # Handle cases where original_pixmap is None or scale is invalid
            self._current_pixmap = None
            self.update() # Request a repaint even if the pixmap is cleared

    def paintEvent(self, event):
        """Draw the current scaled pixmap centered in the label."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._current_pixmap:
            # Calculate the position to draw the pixmap to center it in the label
            label_rect = self.contentsRect()
            pixmap_size = self._current_pixmap.size()

            # Calculate the top-left corner for drawing the pixmap to center it
            x = label_rect.x() + (label_rect.width() - pixmap_size.width()) // 2 + self._total_pan_offset.x()
            y = label_rect.y() + (label_rect.height() - pixmap_size.height()) // 2 + self._total_pan_offset.y()

            # Draw the scaled pixmap at the calculated position
            painter.drawPixmap(int(x), int(y), self._current_pixmap)

            # Draw info box if visible
            if self._info_box_visible and self._info_box_data and self._info_box_position:
                # Draw a subtle shadow
                shadow_offset = 2
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 50))  # Darker shadow for dark theme
                shadow_rect = self._info_box_position.adjusted(shadow_offset, shadow_offset, shadow_offset, shadow_offset)
                painter.drawRect(shadow_rect)
                
                # Draw the main box
                painter.setPen(QPen(QColor(60, 60, 60), 1))  # Dark border
                painter.setBrush(QColor(40, 44, 52))  # Atom-like dark background
                painter.drawRect(self._info_box_position)
                
                # Set up text style
                painter.setPen(QColor(171, 178, 191))  # Atom-like text color
                font = QFont("Segoe UI", 9)
                font.setBold(True)
                painter.setFont(font)
                
                # Format and draw the info text
                text = self._format_info_text(self._info_box_data)
                text_rect = self._info_box_position.adjusted(10, 10, -10, -10)
                painter.drawText(text_rect, 
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                               text)
        else:
            # If no pixmap is set, call the base class paintEvent to draw background etc.
            super().paintEvent(event)

        self.update() # Request a repaint after applying scale due to resize

    def _format_info_text(self, data):
        """Format the hover info data into a readable string."""
        text = []
        if 'type' in data:
            text.append(f"📋 {data['type'].upper()}")  # Add icon and make type uppercase
        if 'confidence' in data:
            text.append(f"🎯 {data['confidence']:.2f}%")  # Add icon and percentage
        if 'text' in data:
            text.append(f"📝 {data['text']}")  # Add icon for text
        if 'associated_text' in data and data['associated_text']:
            text.append("🔗 Associated Text:")  # Add icon for associated text
            for assoc in data['associated_text']:
                text.append(f"   • {assoc.get('text', 'N/A')}")  # Indent with bullet point
        return "\n".join(text)

    def _calculate_info_box_position(self, bbox, mouse_pos):
        """Calculate the optimal position for the info box."""
        if not self._current_pixmap:
            return None

        # Get the image position in the label
        label_rect = self.contentsRect()
        pixmap_size = self._current_pixmap.size()
        image_x = label_rect.x() + (label_rect.width() - pixmap_size.width()) // 2 + self._total_pan_offset.x()
        image_y = label_rect.y() + (label_rect.height() - pixmap_size.height()) // 2 + self._total_pan_offset.y()

        # Scale the bbox coordinates
        x1, y1, x2, y2 = bbox
        scaled_x1 = int(x1 * self._current_scale) + image_x
        scaled_y1 = int(y1 * self._current_scale) + image_y
        scaled_x2 = int(x2 * self._current_scale) + image_x
        scaled_y2 = int(y2 * self._current_scale) + image_y

        # Calculate info box size (approximate)
        box_width = 200
        box_height = 100

        # Calculate available space in each direction
        space_right = label_rect.right() - scaled_x2
        space_left = scaled_x1 - label_rect.left()
        space_top = scaled_y1 - label_rect.top()
        space_bottom = label_rect.bottom() - scaled_y2

        # Determine position
        if space_right >= box_width:
            # Place to the right
            x = scaled_x2 + 5
            y = scaled_y1
        elif space_left >= box_width:
            # Place to the left
            x = scaled_x1 - box_width - 5
            y = scaled_y1
        else:
            # Place below
            x = scaled_x1
            y = scaled_y2 + 5

        # Adjust y position if box would go below the image
        if y + box_height > label_rect.bottom():
            y = label_rect.bottom() - box_height - 5

        # Ensure box stays within label bounds
        x = max(label_rect.left() + 5, min(x, label_rect.right() - box_width - 5))
        y = max(label_rect.top() + 5, min(y, label_rect.bottom() - box_height - 5))

        return QRectF(x, y, box_width, box_height)

    def set_info_box(self, data=None, visible=False):
        """Set the info box data and visibility."""
        self._info_box_data = data
        self._info_box_visible = visible
        if data and visible and self._highlight_bbox:
            self._info_box_position = self._calculate_info_box_position(
                self._highlight_bbox,
                self.mapFromGlobal(self.cursor().pos())
            )
        else:
            self._info_box_position = None
        self.update()

    # Keep resizeEvent to allow image to scale with window resizing
    def resizeEvent(self, event):
        # Reapply the current scale when the label is resized (e.g., tab switch)
        # This ensures the image maintains its size relative to the original
        if self._original_pixmap:
            self._apply_scale()

        super().resizeEvent(event)

    # Re-implemented wheelEvent to enable mouse zoom
    def wheelEvent(self, event: QWheelEvent):
        # Zoom in/out based on wheel movement
        zoom_factor = 1.15 # Adjust zoom sensitivity
        # Get mouse position relative to the image
        mouse_pos = event.position()
        if self._current_pixmap:
            # Calculate the position of the image within the label
            label_rect = self.contentsRect()
            pixmap_size = self._current_pixmap.size()
            x_offset = label_rect.x() + (label_rect.width() - pixmap_size.width()) // 2
            y_offset = label_rect.y() + (label_rect.height() - pixmap_size.height()) // 2

            # Calculate mouse position relative to the image
            mouse_x = mouse_pos.x() - x_offset
            mouse_y = mouse_pos.y() - y_offset

            # Only zoom if mouse is over the image
            if 0 <= mouse_x < pixmap_size.width() and 0 <= mouse_y < pixmap_size.height():
                # Calculate the point in the original image coordinates
                original_x = mouse_x / self._current_scale
                original_y = mouse_y / self._current_scale

                # Store the old scale for calculating the new position
                old_scale = self._current_scale

                # Apply zoom
                if event.angleDelta().y() > 0: # Scroll up - zoom in
                    self._current_scale *= zoom_factor
                else: # Scroll down - zoom out
                    self._current_scale /= zoom_factor

                # Calculate the new position to keep the point under mouse cursor
                new_x = original_x * self._current_scale
                new_y = original_y * self._current_scale

                # Calculate the offset needed to keep the point under the mouse
                x_offset = mouse_x - new_x
                y_offset = mouse_y - new_y

                # Apply the new scale and position
                self._apply_scale()
                return

        # If mouse is not over the image, zoom from center
        if event.angleDelta().y() > 0: # Scroll up - zoom in
            self._current_scale *= zoom_factor
        else: # Scroll down - zoom out
            self._current_scale /= zoom_factor

        self._apply_scale() # Apply the new scale

    def mousePressEvent(self, event):
        """Handle mouse press events for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self._last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for panning and hover."""
        if self._is_panning and self._last_mouse_pos is not None:
            # Calculate the movement delta
            delta = event.position() - self._last_mouse_pos
            self._total_pan_offset += delta
            self._last_mouse_pos = event.position()
            self.update()
            return

        # Original hover coordinate calculation
        if self._current_pixmap and self._original_pixmap:
            scaled_pixmap_size = self._current_pixmap.size()
            label_size = self.size()
            original_pixmap_size = self._original_pixmap.size()

            # Calculate the size of the scaled image while maintaining aspect ratio
            image_width_scaled = original_pixmap_size.width() * self._current_scale
            image_height_scaled = original_pixmap_size.height() * self._current_scale

            # Calculate centering offsets for the scaled image within the label
            x_offset = max(0, (label_size.width() - image_width_scaled) // 2) + self._total_pan_offset.x()
            y_offset = max(0, (label_size.height() - image_height_scaled) // 2) + self._total_pan_offset.y()

            # Mouse position relative to the top-left of the scaled image content
            mouse_x_scaled = event.position().x() - x_offset
            mouse_y_scaled = event.position().y() - y_offset

            # Check if mouse is within the bounds where the scaled pixmap is actually drawn
            if 0 <= mouse_x_scaled < image_width_scaled and 0 <= mouse_y_scaled < image_height_scaled:
                # Translate scaled mouse position back to original image coordinates
                original_x = int(mouse_x_scaled / self._current_scale)
                original_y = int(mouse_y_scaled / self._current_scale)
                # Emit a signal with the original image coordinates
                self.image_hovered_coords.emit(QPoint(original_x, original_y))
            else:
                # If mouse is outside the image, emit a signal with invalid coordinates
                self.image_hovered_coords.emit(QPoint(-1, -1))

        super().mouseMoveEvent(event)

    def set_highlight(self, bbox=None, color=QColor(255, 0, 0), thickness=2):
        """Sets the bounding box to be highlighted in original image coordinates."""
        self._highlight_bbox = bbox
        self._highlight_color = color
        self._highlight_thickness = thickness
        self._apply_scale() # Redraw the image with the new highlight

    # Add a method to get the current scale for external use if needed
    def get_current_scale(self):
         return self._current_scale

    def zoom_in(self):
        """Increase the current scale to zoom in."""
        zoom_factor = 1.15 # Same zoom sensitivity as wheel
        self._current_scale *= zoom_factor
        self._apply_scale()

    def zoom_out(self):
        """Decrease the current scale to zoom out."""
        zoom_factor = 1.15 # Same zoom sensitivity as wheel
        self._current_scale /= zoom_factor
        self._apply_scale()


def draw_annotations(image_cv, yolo_elements=None, ocr_text_blocks=None, draw_yolo_associated_text=False, draw_element_type=False):
    """Draws YOLO and/or OCR annotations on a copy of the OpenCV image."""
    # Create a deep copy to draw on without modifying the original image
    annotated_image_cv = image_cv.copy()

    # Convert to QImage to use QPainter for drawing
    height, width, channel = annotated_image_cv.shape
    bytes_per_line = 3 * width
    # Ensure the image is in a format QImage can handle directly, e.g., BGR or RGB
    # OpenCV reads as BGR, QImage expects RGB for Format_RGB888, BGR for Format_BGR888
    # We use rgbSwapped() when creating the QImage for consistent RGB handling if needed,
    # but for drawing with QPainter on a BGR image, Format_BGR888 might be appropriate.
    # Let's stick to the current approach of creating BGR QImage and using painter.setPen/setBrush
    # The colors we set are QColor objects, which QPainter handles correctly regardless of the underlying QImage format.
    q_image = QImage(annotated_image_cv.data, width, height, bytes_per_line, QImage.Format.Format_BGR888).rgbSwapped() # Swapped to RGB for consistent color handling


    painter = QPainter(q_image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing) # For smoother drawing

    font = QFont("Arial", 8)  # Reduced font size from 12 to 8
    painter.setFont(font)

    # Define colors for annotations
    element_bbox_color = QColor(0, 255, 0) # Green for Element (YOLO) bounding boxes
    text_bbox_color = QColor(0, 255, 0)    # Green for Text (OCR) bounding boxes
    yolo_text_color = QColor(0, 0, 255)    # Blue for YOLO element text
    ocr_text_color = QColor(255, 0, 255)   # Magenta for PaddleOCR text
    element_type_color = QColor(0, 0, 255) # Blue for Element type text

    # Draw YOLO bounding boxes
    if yolo_elements:
        painter.setPen(QPen(element_bbox_color, 2)) # Green pen, 2px thick for element boxes
        # Set brush to NoBrush for drawing outlines only
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for element in yolo_elements:
            x1, y1, x2, y2 = element['bbox']
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)

            # Draw element type text if requested
            if draw_element_type:
                 label_text = element.get('type', 'N/A')
                 # Simple text placement above the box
                 text_rect = QRectF(x1, y1 - 20, x2 - x1, 20) # 20px high rect above the bbox
                 painter.setPen(QPen(element_type_color)) # Blue text color
                 # Use AlignBottom to position text just above the box
                 painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, label_text)

            # If requested, draw associated text for YOLO elements
            if draw_yolo_associated_text and 'associated_text' in element and element['associated_text']:
                # Draw associated text blocks for this element
                for assoc_text_block in element['associated_text']:
                    text_x1, text_y1, text_x2, text_y2 = assoc_text_block['bbox']
                    text_rect = QRectF(text_x1, text_y1, text_x2 - text_x1, text_y2 - text_y1)
                    
                    # Draw text content in blue above the element
                    text_content = assoc_text_block.get('text', 'N/A')
                    text_x = x1 # Draw text starting at the left edge of the element bbox
                    text_y = y1 - 5 # Draw text slightly above the element bbox
                    painter.setPen(QPen(yolo_text_color)) # Blue text color
                    painter.drawText(QPointF(text_x, text_y), text_content)

    # Draw OCR bounding boxes and text
    if ocr_text_blocks:
        # Use a thinner pen for OCR text boxes
        painter.setPen(QPen(text_bbox_color, 1)) # Green pen, 1px thick for text boxes
        # Set brush to NoBrush for drawing outlines only
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for text_block in ocr_text_blocks:
            x1, y1, x2, y2 = text_block['bbox']
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)
            # Draw text
            text_content = text_block.get('text', 'N/A')
            # Draw text at top-right of the box in magenta
            text_x = x2 - 5  # 5 pixels from right edge
            text_y = y1 - 5  # 5 pixels above top edge
            painter.setPen(QPen(ocr_text_color)) # Magenta text color
            # Right-align the text
            text_rect = QRectF(x1, y1 - 20, x2 - x1, 20)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, text_content)


    painter.end()
    return QPixmap.fromImage(q_image)