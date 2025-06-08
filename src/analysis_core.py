import cv2
import numpy as np
import json
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PyQt6.QtCore import QRectF # Import QRectF for IoU calculation

class AnalysisCore:
    def __init__(self, yolo_model_path, ocr_params):
        self.yolo_model = None
        self.yolo_class_names = None
        self.ocr_model = None

        try:
            # Load YOLO model locally
            print(f"Loading YOLO model from {yolo_model_path}")
            self.yolo_model = YOLO(yolo_model_path)
            self.yolo_class_names = self.yolo_model.names
            print("YOLO model loaded.")
        except Exception as yolo_e:
            print(f"Error loading YOLO model from {yolo_model_path}: {yolo_e}")
            self.yolo_model = None
            self.yolo_class_names = None


        try:
             print("Initializing PaddleOCR model...")
             self.ocr_model = PaddleOCR(**ocr_params)
             print("PaddleOCR model initialized.")
        except Exception as ocr_e:
             print(f"Error initializing PaddleOCR: {ocr_e}")
             self.ocr_model = None

        if self.yolo_model and self.ocr_model:
             print("All models loaded successfully in AnalysisCore.")
        else:
             print("One or more models failed to load in AnalysisCore.")


    def run_analysis(self, image_cv):
        """Runs YOLO and PaddleOCR inference and returns raw results."""
        if self.yolo_model is None or self.ocr_model is None:
             print("Models not loaded. Cannot run analysis.")
             return None, None # Return empty results

        # --- YOLO Detection ---
        print("Running YOLO inference...")
        yolo_results = []
        try:
            results_yolo = self.yolo_model.predict(source=image_cv, imgsz=640, conf=0.25, verbose=False)
            if results_yolo and len(results_yolo) > 0:
                result = results_yolo[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = box.cls[0].item()
                        class_name = self.yolo_class_names[int(cls)] if self.yolo_class_names and int(cls) < len(self.yolo_class_names) else f"unknown_{int(cls)}"

                        yolo_results.append({
                            "type": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                            "associated_text": []
                        })
            print(f"Total YOLO elements detected: {len(yolo_results)}")
        except Exception as e:
             print(f"Error during YOLO inference: {e}")
             # Continue with empty YOLO results if inference fails


        # --- PaddleOCR Detection ---
        print("Running PaddleOCR inference using predict()...")
        ocr_results = []
        try:
            raw_ocr_results = self.ocr_model.predict(image_cv)
            if raw_ocr_results:
                result_obj = raw_ocr_results[0]
                if result_obj and 'rec_polys' in result_obj and 'rec_texts' in result_obj and 'rec_scores' in result_obj:
                    polys = result_obj['rec_polys']
                    texts = result_obj['rec_texts']
                    scores = result_obj['rec_scores']

                    print(f"Processing OCR results from predict(). Found {len(polys)} text detections.")

                    if len(polys) == len(texts) == len(scores):
                         for i in range(len(polys)):
                            bbox_points = polys[i]
                            text = texts[i]
                            confidence = scores[i]

                            if text and text.strip():
                                 if (isinstance(bbox_points, list) or isinstance(bbox_points, np.ndarray)) and len(bbox_points) == 4 and all(isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2 for p in bbox_points):
                                      if isinstance(bbox_points, np.ndarray):
                                           bbox_points = bbox_points.tolist()

                                      x_coords = [p[0] for p in bbox_points]
                                      y_coords = [p[1] for p in bbox_points]
                                      x1_ocr, y1_ocr, x2_ocr, y2_ocr = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

                                      ocr_results.append({
                                          'text': text.strip(),
                                          'bbox': [int(x1_ocr), int(y1_ocr), int(x2_ocr), int(y2_ocr)],
                                          'confidence': float(confidence)
                                      })
                                 else:
                                      print(f"Warning: Skipping detection at index {i} due to unexpected bbox_points format or size: {bbox_points}")
                    else:
                         print(f"Warning: Mismatch in lengths of polys ({len(polys)}), texts ({len(texts)}), and scores ({len(scores)}).")
                else:
                    print("Warning: Raw OCR predict results object does not contain expected keys.")
            else:
                print("No raw OCR predict results returned.")

            print(f"Total valid OCR text blocks (after filtering and parsing from predict output): {len(ocr_results)}")

        except Exception as e:
            print(f"Error running PaddleOCR predict(): {e}")
            # Continue with empty OCR results if inference fails
            pass

        return yolo_results, ocr_results

    def associate_results(self, yolo_results, ocr_results):
        """Associates OCR results with YOLO elements and generates the final structured data."""
        if yolo_results is None and ocr_results is None:
             return []

        json_output_elements = json.loads(json.dumps(yolo_results)) if yolo_results else [] # Simple deep copy
        ocr_results_copy = json.loads(json.dumps(ocr_results)) if ocr_results else [] # Simple deep copy for processing


        associated_ocr_indices = set()

        # Sort YOLO elements and OCR results spatially for more consistent processing order
        json_output_elements.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        ocr_results_copy.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))

        # Add index to YOLO elements
        for idx, element in enumerate(json_output_elements):
            element['index'] = idx

        for i, ocr_res in enumerate(ocr_results_copy):
            ocr_bbox_rect = QRectF(*ocr_res['bbox'])

            best_match_element_index = -1
            best_iou = 0.0
            contained_in_element_index = -1 # To store the index of element that fully contains this text

            for elem_index, element in enumerate(json_output_elements):
                element_bbox_rect = QRectF(*element['bbox'])

                # Check for full containment of the OCR bbox within the element bbox
                if element_bbox_rect.contains(ocr_bbox_rect):
                     contained_in_element_index = elem_index
                     break # Prioritize containment, stop searching for this OCR

                # If not contained, calculate IoU for overlap check
                intersection_rect = element_bbox_rect.intersected(ocr_bbox_rect)
                if not intersection_rect.isEmpty():
                     intersection_area = intersection_rect.width() * intersection_rect.height()
                     ocr_area = ocr_bbox_rect.width() * ocr_bbox_rect.height()
                     element_area = element_bbox_rect.width() * element_bbox_rect.height()

                     # Calculate IoU
                     union_area = ocr_area + element_area - intersection_area
                     iou = intersection_area / union_area if union_area > 0 else 0

                     iou_threshold = 0.3 # Adjusted IoU threshold slightly
                     if contained_in_element_index == -1 and iou > iou_threshold and iou > best_iou:
                          best_iou = iou
                          best_match_element_index = elem_index

            associated_element_index = contained_in_element_index if contained_in_element_index != -1 else best_match_element_index

            if associated_element_index != -1:
                json_output_elements[associated_element_index]['associated_text'].append(ocr_res)
                # Find the index in the original ocr_results list
                try:
                    original_ocr_index = ocr_results.index(next(item for item in ocr_results if item['bbox'] == ocr_res['bbox'] and item['text'] == ocr_res['text']))
                    associated_ocr_indices.add(original_ocr_index)
                except (ValueError, StopIteration):
                    print(f"Warning: Could not find original index for OCR result: {ocr_res}")


        final_json_output_data = []
        final_json_output_data.extend(json_output_elements)

        standalone_text_elements = []
        if ocr_results: # Ensure ocr_results is not None
            for i, ocr_res in enumerate(ocr_results):
                if i not in associated_ocr_indices:
                     standalone_text_elements.append({
                        "type": "text",
                        "confidence": ocr_res['confidence'],
                        "bbox": ocr_res['bbox'],
                        "text": ocr_res['text'],
                        "index": len(final_json_output_data) + len(standalone_text_elements) # Add index for standalone text elements
                    })

        standalone_text_elements.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        final_json_output_data.extend(standalone_text_elements)

        def sort_key(item):
             type_order = 0 if item.get('type') != 'text' else 1
             bbox = item.get('bbox', [0, 0, 0, 0])
             return (type_order, bbox[1], bbox[0])

        final_json_output_data.sort(key=sort_key)

        # Reassign indices after final sorting to ensure they are sequential
        for idx, item in enumerate(final_json_output_data):
            item['index'] = idx

        print(f"Total items in final JSON data: {len(final_json_output_data)}")

        return final_json_output_data