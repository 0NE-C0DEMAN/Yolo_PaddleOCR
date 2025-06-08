import json
import os
import google.generativeai as genai
from typing import Dict, List, Any
from datetime import datetime
from collections import deque

class GeminiHandler:
    def __init__(self, config_path: str, max_history: int = 10):
        self.config_path = config_path
        self.history = []
        self.conversation_history = deque(maxlen=max_history)
        self.current_analysis_data = None
        self.current_image_name = None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('gemini_api_key')
                if api_key:
                    genai.configure(api_key=api_key)
                    # Get model configuration from config file
                    model_name = config.get('model', 'gemini-pro')
                    temperature = config.get('temperature', 0.7)
                    max_tokens = config.get('max_output_tokens', 1024)
                    
                    # Initialize the model with configuration
                    self.model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config={
                            'temperature': temperature,
                            'max_output_tokens': max_tokens
                        }
                    )
                    print(f"Initialized Gemini model: {model_name}")
                else:
                    print("Warning: No Gemini API key found in config file")
        except Exception as e:
            print(f"Error loading Gemini config: {e}")
            raise  # Re-raise the exception to handle it in the calling code

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading Gemini config: {e}")
            return {}

    def _setup_gemini(self):
        try:
            self.model = genai.GenerativeModel(
                model_name=self.config.get('model', 'gemini-pro'),
                generation_config={
                    'temperature': self.config.get('temperature', 0.7),
                    'max_output_tokens': self.config.get('max_output_tokens', 1024),
                }
            )
        except Exception as e:
            print(f"Error setting up Gemini: {e}")
            self.model = None

    def _format_conversation_history(self) -> str:
        """Format the conversation history into a readable string."""
        if not self.conversation_history:
            return "No previous conversation history."

        formatted_history = "Previous conversation:\n"
        for entry in self.conversation_history:
            timestamp = entry.get('timestamp', 'Unknown time')
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')
            formatted_history += f"[{timestamp}] {role.capitalize()}: {content}\n"
        return formatted_history

    def _create_context_aware_prompt(self, user_query: str) -> str:
        """Create a prompt that includes conversation history and current context."""
        # Start with the system role and current analysis data
        prompt = f"""You are an AI assistant analyzing UI elements and text from an image. 
        You have access to the current analysis data and previous conversation history.
        
        Current Image: {self.current_image_name or 'Not specified'}
        
        Current Analysis Data:
        {json.dumps(self.current_analysis_data, indent=2) if self.current_analysis_data else 'No analysis data available'}

        {self._format_conversation_history()}

        Current User Query: {user_query}

        Please provide a clear and concise answer based on both the current analysis data and the conversation history.
        Follow these formatting guidelines:
        1. Use clear, concise language
        2. Format lists with proper spacing and alignment
        3. When listing elements, use a clean format like:
           - Element 1: [description]
           - Element 2: [description]
        4. Avoid raw JSON or technical details unless specifically requested
        5. Use markdown formatting for better readability
        6. Keep coordinates and technical details in a separate line or section if needed
        7. Focus on the most relevant information first

        If the query cannot be answered from the available data, say so politely.
        Maintain context from previous interactions when relevant."""

        return prompt

    def generate_response(self, user_query: str, analysis_data: List[Dict[str, Any]], image_name: str = None) -> str:
        if not self.model:
            return "Error: Gemini model not properly initialized. Please check your API key and configuration."

        try:
            # Update current context
            self.current_analysis_data = analysis_data
            self.current_image_name = image_name

            # Create context-aware prompt
            prompt = self._create_context_aware_prompt(user_query)

            # Generate response
            response = self.model.generate_content(prompt)
            
            # Format the response text for better readability
            formatted_response = self._format_response(response.text)
            
            # Add to conversation history with timestamp
            self.conversation_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": "user",
                "content": user_query,
                "image": image_name
            })
            self.conversation_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "role": "assistant",
                "content": formatted_response,
                "image": image_name
            })

            return formatted_response

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg

    def _format_response(self, response_text: str) -> str:
        """Format the response text for better readability."""
        # Split the response into lines
        lines = response_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Format list items
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.')):
                # Add proper spacing for list items
                formatted_lines.append(f"\n{line.strip()}")
            # Format text elements
            elif 'Text:' in line and 'bbox:' in line:
                # Split into text and bbox parts
                text_part = line.split('bbox:')[0].strip()
                bbox_part = line.split('bbox:')[1].strip()
                # Format as a clean list item
                formatted_lines.append(f"\n- {text_part}")
                # Add bbox info on next line with proper indentation
                formatted_lines.append(f"  Location: {bbox_part}")
            else:
                # Regular text, just add proper spacing
                formatted_lines.append(line.strip())
        
        # Join lines with proper spacing
        return '\n'.join(formatted_lines)

    def clear_history(self):
        """Clear the conversation history and current context."""
        self.conversation_history.clear()
        self.current_analysis_data = None
        self.current_image_name = None

    def get_recent_context(self, num_exchanges: int = 3) -> List[Dict]:
        """Get the most recent conversation exchanges."""
        return list(self.conversation_history)[-num_exchanges*2:] if self.conversation_history else [] 