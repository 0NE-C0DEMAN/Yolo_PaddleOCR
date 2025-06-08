import json
import os

class DataManager:
    def __init__(self, output_dir):
        self._analysis_cache = {}
        self.output_dir = output_dir
        # Ensure cache directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def get_cached_analysis(self, image_path):
        """Retrieves cached analysis data for a given image path."""
        return self._analysis_cache.get(image_path)

    def cache_analysis(self, image_path, analysis_data, yolo_results, ocr_results):
        """Caches analysis results in memory and saves to a JSON file."""
        if image_path:
             cached_data = {
                 'analysis_data': analysis_data,
                 'yolo_results': yolo_results,
                 'ocr_results': ocr_results
             }
             self._analysis_cache[image_path] = cached_data
             print(f"Analysis results cached in memory for {image_path}")
             self._save_to_json(image_path, analysis_data)

    def _save_to_json(self, image_path, analysis_data):
        """Saves the analysis data to a JSON file."""
        if image_path and analysis_data is not None:
            try:
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                json_output_path = os.path.join(self.output_dir, f'{base_filename}_output.json')
                with open(json_output_path, 'w') as f:
                    json.dump(analysis_data, f, indent=4)
                print(f"JSON output saved successfully to {json_output_path}")
            except Exception as e:
                print(f"Error saving JSON to file {json_output_path}: {e}")

    def load_analysis_from_json(self, image_path):
        """Loads analysis data from a JSON file if it exists."""
        if image_path:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            json_output_path = os.path.join(self.output_dir, f'{base_filename}_output.json')
            if os.path.exists(json_output_path):
                try:
                    with open(json_output_path, 'r') as f:
                        analysis_data = json.load(f)
                    print(f"Analysis data loaded from JSON file: {json_output_path}")
                    return analysis_data
                except Exception as e:
                    print(f"Error loading JSON from file {json_output_path}: {e}")
        return None

    def clear_cache(self):
        """Clears the in-memory cache."""
        self._analysis_cache = {}
        print("Analysis cache cleared.")
