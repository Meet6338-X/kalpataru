"""
AI Assistant Service for Kalpataru.
Provides intelligent chat capabilities with image understanding and RAG-like context.
Uses OpenRouter API for model access.
Integrates with ML disease detection model for enhanced image analysis.
"""

import os
import base64
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path
from io import BytesIO
import requests

from utils.logger import setup_logger
from config import settings

logger = setup_logger(__name__)

# Path to disease info JSON
DISEASE_INFO_PATH = Path(__file__).parent.parent / "ml" / "disease_info.json"


class AgriculturalKnowledgeBase:
    """
    A simple RAG-like knowledge base for agricultural context.
    Provides relevant context based on user queries.
    Includes disease information from ML model for enhanced responses.
    """
    
    def __init__(self):
        self.knowledge_chunks = self._load_knowledge()
        self.disease_info = self._load_disease_info()
    
    def _load_disease_info(self) -> Dict[str, Any]:
        """Load disease information from JSON file for ML model integration."""
        if DISEASE_INFO_PATH.exists():
            try:
                with open(DISEASE_INFO_PATH, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded disease info for {len(data.get('diseases', {}))} diseases")
                    return data.get('diseases', {})
            except Exception as e:
                logger.warning(f"Could not load disease info: {e}")
        return {}
    
    def _load_knowledge(self) -> List[Dict[str, str]]:
        """Load agricultural knowledge chunks for RAG."""
        knowledge_chunks = [
            {
                "topic": "crop_recommendation",
                "content": """
                Crop Recommendation Guidelines:
                - Rice: Requires high rainfall (200-300cm), temperature 20-35°C, clay or loamy soil, pH 5.5-6.5
                - Wheat: Needs moderate rainfall (50-100cm), temperature 10-25°C, well-drained loamy soil, pH 6-7
                - Maize: Requires 60-100cm rainfall, temperature 18-32°C, well-drained fertile soil
                - Cotton: Needs 60-100cm rainfall, temperature 20-30°C, black cotton soil ideal
                - Sugarcane: Requires 150-200cm rainfall, temperature 20-35°C, deep loamy soil
                - Groundnut: Needs 50-100cm rainfall, temperature 20-30°C, sandy loam soil
                """
            },
            {
                "topic": "disease_management",
                "content": """
                Plant Disease Management:
                - Early Blight: Caused by Alternaria fungus. Symptoms include target-like lesions on leaves. 
                  Treatment: Remove infected leaves, apply copper-based fungicides, ensure good air circulation.
                - Late Blight: Caused by Phytophthora infestans. Water-soaked lesions that turn brown.
                  Treatment: Remove infected plants, apply fungicides preventively, avoid overhead irrigation.
                - Powdery Mildew: White powdery growth on leaves. Common in humid conditions.
                  Treatment: Improve air circulation, apply sulfur-based fungicides, avoid excess nitrogen.
                - Bacterial Spot: Small water-soaked spots on leaves and fruit.
                  Treatment: Use disease-free seeds, copper sprays, avoid working with wet plants.
                """
            },
            {
                "topic": "fertilizer_management",
                "content": """
                Fertilizer Management Guidelines:
                - Nitrogen (N): Promotes leaf growth and green color. Deficiency causes yellowing of older leaves.
                  Sources: Urea (46% N), Ammonium Sulfate (21% N), DAP (18% N)
                - Phosphorus (P): Essential for root development and flowering. Deficiency causes purple coloring.
                  Sources: DAP (46% P2O5), Single Super Phosphate (16% P2O5)
                - Potassium (K): Important for disease resistance and water regulation.
                  Sources: MOP (60% K2O), Potassium Sulfate (50% K2O)
                - Application timing: Apply basal dose at sowing, top dressing at critical growth stages.
                """
            },
            {
                "topic": "irrigation",
                "content": """
                Irrigation Best Practices:
                - Critical growth stages for irrigation vary by crop
                - Rice: Maintain standing water during vegetative and reproductive stages
                - Wheat: Crown root initiation (21 days), tillering, jointing, heading stages critical
                - Cotton: Flowering and boll development stages most sensitive to water stress
                - Drip irrigation: Most efficient, saves 30-50% water, suitable for vegetables and fruits
                - Sprinkler irrigation: Good for close-spaced crops, can be used for frost protection
                - Schedule irrigation based on soil moisture, not fixed calendar dates
                """
            },
            {
                "topic": "soil_health",
                "content": """
                Soil Health Management:
                - Alluvial soil: Highly fertile, good for wheat, rice, sugarcane. pH 6.5-8.4
                - Black soil: Rich in lime, iron, magnesia. Good for cotton, cereals. pH 7.0-8.5
                - Red soil: Poor in nitrogen, phosphorus, humus. Needs organic matter. pH 5.5-7.5
                - Laterite soil: Rich in iron and aluminum, poor in fertility. Good for tea, coffee. pH 4.5-6.5
                - Sandy soil: Low fertility, quick drainage. Good for groundnut, watermelon. pH 5.5-7.0
                - Clay soil: High fertility, poor drainage. Good for rice, pulses. pH 6.0-7.5
                """
            },
            {
                "topic": "pest_management",
                "content": """
                Integrated Pest Management (IPM):
                - Prevention: Use resistant varieties, crop rotation, healthy seeds
                - Monitoring: Regular field scouting, pheromone traps, yellow sticky traps
                - Biological control: Encourage natural enemies (ladybugs, lacewings, parasitic wasps)
                - Chemical control: Use as last resort, choose selective pesticides, follow waiting periods
                - Common pests: Aphids, whiteflies, bollworms, stem borers, fruit flies
                - Organic options: Neem oil, Bacillus thuringiensis (Bt), pyrethrum
                """
            },
            {
                "topic": "weather_impact",
                "content": """
                Weather Impact on Agriculture:
                - Temperature: Affects germination, flowering, fruit set, and pest activity
                - Rainfall: Critical for rainfed crops, excess causes waterlogging and disease
                - Humidity: High humidity favors fungal diseases, affects pollination
                - Wind: Strong winds cause lodging, spread pests and diseases
                - Frost: Damages tender crops, affects fruit quality
                - Drought: Causes water stress, reduces yield, affects quality
                - Weather-based advisories help farmers plan activities optimally
                """
            },
            {
                "topic": "organic_farming",
                "content": """
                Organic Farming Practices:
                - Soil fertility: Compost, vermicompost, green manure, biofertilizers
                - Pest control: Neem, cow urine, biological pesticides, trap crops
                - Disease management: Trichoderma, Pseudomonas, copper sprays
                - Weed management: Mulching, manual weeding, cover crops
                - Certification: Requires 3-year transition period, documentation essential
                - Benefits: Premium prices, soil health, environmental sustainability
                """
            }
        ]
        
        # Add disease-specific knowledge from ML module
        disease_knowledge = self._load_disease_knowledge()
        knowledge_chunks.extend(disease_knowledge)
        
        return knowledge_chunks
    
    def _load_disease_knowledge(self) -> List[Dict[str, str]]:
        """Load disease-specific knowledge chunks from disease_info.json."""
        disease_chunks = []
        
        if DISEASE_INFO_PATH.exists():
            try:
                with open(DISEASE_INFO_PATH, 'r') as f:
                    data = json.load(f)
                    diseases = data.get('diseases', {})
                    
                    for disease_key, info in diseases.items():
                        # Create a knowledge chunk for each disease
                        chunk = {
                            "topic": f"disease_{disease_key}",
                            "content": f"""
                            Disease: {info.get('disease', 'Unknown')}
                            Plant: {info.get('plant', 'Unknown')}
                            Severity: {info.get('severity', 'Unknown')}
                            Description: {info.get('description', 'N/A')}
                            Symptoms: {info.get('symptoms', 'N/A')}
                            Treatment: {info.get('treatment', 'N/A')}
                            Prevention: {info.get('prevention', 'N/A')}
                            """
                        }
                        disease_chunks.append(chunk)
                    
                    logger.info(f"Added {len(disease_chunks)} disease knowledge chunks to RAG")
            except Exception as e:
                logger.warning(f"Could not load disease knowledge chunks: {e}")
        
        return disease_chunks
    
    def get_disease_info(self, disease_key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific disease."""
        return self.disease_info.get(disease_key)
    
    def get_relevant_context(self, query: str, max_chunks: int = 3) -> str:
        """
        Get relevant knowledge chunks based on query keywords.
        Simple keyword matching for RAG-like functionality.
        """
        query_lower = query.lower()
        relevant_chunks = []
        
        # Keyword mapping for topics
        topic_keywords = {
            "crop_recommendation": ["crop", "grow", "plant", "cultivate", "rice", "wheat", "maize", "cotton", "sugarcane", "recommendation"],
            "disease_management": ["disease", "sick", "infection", "blight", "mildew", "spot", "rot", "fungus", "bacterial", "virus"],
            "fertilizer_management": ["fertilizer", "npk", "nitrogen", "phosphorus", "potassium", "urea", "dap", "nutrient", "deficiency"],
            "irrigation": ["irrigation", "water", "drip", "sprinkler", "moisture", "drought", "rainfall"],
            "soil_health": ["soil", "ph", "alluvial", "black soil", "red soil", "clay", "sandy", "laterite", "fertility"],
            "pest_management": ["pest", "insect", "aphid", "worm", "beetle", "bug", "ipm", "pesticide"],
            "weather_impact": ["weather", "temperature", "rain", "humidity", "wind", "frost", "climate"],
            "organic_farming": ["organic", "natural", "compost", "vermicompost", "biofertilizer", "sustainable"]
        }
        
        # Score each chunk based on keyword matches
        chunk_scores = []
        for chunk in self.knowledge_chunks:
            topic = chunk["topic"]
            keywords = topic_keywords.get(topic, [])
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                chunk_scores.append((score, chunk))
        
        # Sort by score and take top chunks
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        relevant_chunks = [chunk for _, chunk in chunk_scores[:max_chunks]]
        
        if relevant_chunks:
            context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
            return context
        
        return ""


class AIAssistant:
    """
    AI Assistant with image understanding and RAG capabilities.
    Uses OpenRouter API for model access.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", getattr(settings, 'OPENROUTER_API_KEY', ''))
        self.base_url = "https://openrouter.ai/api/v1"
        self.knowledge_base = AgriculturalKnowledgeBase()
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history = 10
        
        # System prompt for agricultural assistant
        self.system_prompt = """You are Kalpataru, an expert agricultural AI assistant specialized in helping farmers with:
- Crop recommendations based on soil, climate, and market conditions
- Plant disease identification and treatment suggestions
- Fertilizer and nutrient management advice
- Irrigation planning and water management
- Pest identification and control measures
- Weather-based agricultural advisories
- Soil health and improvement strategies
- Organic and sustainable farming practices

You provide practical, actionable advice based on scientific agricultural knowledge. 
When analyzing images, you identify plants, diseases, pests, or soil conditions visible.
Always prioritize sustainable and environmentally friendly solutions.
If you're unsure about something, acknowledge the limitation and suggest consulting local agricultural experts.

Respond in a helpful, friendly manner. Use simple language that farmers can easily understand.
When providing recommendations, consider the local context and practical constraints of smallholder farmers."""
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://kalpataru-agri.ai",
            "X-Title": "Kalpataru Agricultural AI"
        }
    
    def _encode_image(self, image_data: Union[bytes, str]) -> str:
        """Encode image to base64 string."""
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        elif isinstance(image_data, str) and image_data.startswith('data:'):
            # Already a data URL
            return image_data
        elif isinstance(image_data, str):
            # Assume it's a URL
            return image_data
        return ""
    
    def _run_disease_model(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Run the ML disease detection model on image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Prediction result dictionary or None if model fails
        """
        try:
            from models.disease.inference import predict_disease
            from io import BytesIO
            
            # Create a file-like object from bytes
            class FileWrapper:
                def __init__(self, data):
                    self._data = data
                    self._pos = 0
                
                def read(self):
                    return self._data
            
            result = predict_disease(FileWrapper(image_data))
            logger.info(f"ML disease prediction: {result.get('disease', 'Unknown')} "
                       f"with {result.get('confidence', 0)*100:.1f}% confidence")
            return result
            
        except Exception as e:
            logger.warning(f"ML disease model prediction failed: {e}")
            return None
    
    def build_disease_context(self, prediction_result: Dict[str, Any]) -> str:
        """
        Build RAG context from disease prediction results.
        
        Args:
            prediction_result: Dictionary with ML model prediction results
            
        Returns:
            Formatted context string for LLM
        """
        disease = prediction_result.get('disease', 'Unknown')
        confidence = prediction_result.get('confidence', 0)
        is_healthy = prediction_result.get('is_healthy', False)
        severity = prediction_result.get('severity', 'Unknown')
        
        # Get detailed disease info from knowledge base
        disease_info = self.knowledge_base.get_disease_info(disease)
        
        # Format top predictions
        top_predictions = prediction_result.get('top_predictions', [])
        top_pred_str = ""
        if top_predictions:
            top_pred_str = "\n".join([
                f"  - {p.get('disease', 'Unknown')}: {p.get('confidence', 0)*100:.1f}%"
                for p in top_predictions[:5]
            ])
        
        context = f"""
ML Model Disease Detection Results:
=========================================
- Predicted Disease: {disease}
- Confidence: {confidence * 100:.1f}%
- Is Healthy: {is_healthy}
- Severity Level: {severity}
"""
        
        # Add detailed disease information if available
        if disease_info:
            context += f"""
Disease Information from Knowledge Base:
-----------------------------------------
- Plant: {disease_info.get('plant', 'N/A')}
- Disease Name: {disease_info.get('disease', 'N/A')}
- Description: {disease_info.get('description', 'N/A')}
- Symptoms: {disease_info.get('symptoms', 'N/A')}
- Treatment: {disease_info.get('treatment', 'N/A')}
- Prevention: {disease_info.get('prevention', 'N/A')}
"""
        
        # Add top predictions if available
        if top_pred_str:
            context += f"""
Top Predictions:
----------------
{top_pred_str}
"""
        
        return context
    
    def _build_messages(
        self, 
        user_message: str, 
        image_data: Optional[Union[bytes, str]] = None,
        image_url: Optional[str] = None,
        include_context: bool = True,
        ml_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build message list for API call.
        
        Args:
            user_message: The user's text message
            image_data: Optional image bytes or base64 string
            image_url: Optional image URL
            include_context: Whether to include RAG context
            ml_context: Optional ML model prediction context
            
        Returns:
            List of message dictionaries for the API
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history
        for msg in self.conversation_history[-self.max_history:]:
            messages.append(msg)
        
        # Build enhanced message with context
        context_parts = []
        
        # Add ML model context first (highest priority)
        if ml_context:
            context_parts.append(ml_context)
        
        # Get relevant context from knowledge base
        if include_context:
            rag_context = self.knowledge_base.get_relevant_context(user_message)
            if rag_context:
                context_parts.append(f"Relevant Agricultural Knowledge:\n{rag_context}")
        
        # Build the final message
        if context_parts:
            enhanced_message = f"""Based on the following context:

{'=' * 50}
{chr(10).join(context_parts)}
{'=' * 50}

User question: {user_message}"""
        else:
            enhanced_message = user_message
        
        # Build user message with optional image
        if image_data or image_url:
            user_content = [
                {"type": "text", "text": enhanced_message}
            ]
            
            if image_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif image_data:
                # Encode image data
                if isinstance(image_data, bytes):
                    b64_image = self._encode_image(image_data)
                    # Assume JPEG format, can be enhanced to detect
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }
                    })
            
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": enhanced_message})
        
        return messages
    
    def chat(
        self,
        message: str,
        image_data: Optional[Union[bytes, str]] = None,
        image_url: Optional[str] = None,
        include_context: bool = True,
        model: str = "nvidia/nemotron-nano-12b-v2-vl:free",
        ml_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a chat message to the AI assistant.
        
        Args:
            message: User's text message
            image_data: Optional image bytes or base64 string
            image_url: Optional image URL
            include_context: Whether to include RAG context
            model: Model to use for generation
            ml_context: Optional ML model prediction context
            
        Returns:
            Response dictionary with assistant's reply
        """
        try:
            messages = self._build_messages(message, image_data, image_url, include_context, ml_context)
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-self.max_history * 2:]
                
                return {
                    "success": True,
                    "response": assistant_message,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"AI API error: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "fallback_response": self._get_fallback_response(message)
                }
                
        except requests.exceptions.Timeout:
            logger.error("AI API timeout")
            return {
                "success": False,
                "error": "Request timeout",
                "fallback_response": self._get_fallback_response(message)
            }
        except Exception as e:
            logger.error(f"AI chat error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_response": self._get_fallback_response(message)
            }
    
    def _get_fallback_response(self, message: str) -> str:
        """Generate a fallback response when API fails."""
        message_lower = message.lower()
        
        # Simple keyword-based responses
        if any(word in message_lower for word in ["disease", "sick", "infection"]):
            return """I'm currently unable to connect to my AI service. For plant disease issues:
1. Remove and destroy infected plant parts
2. Ensure good air circulation around plants
3. Avoid overhead watering
4. Consider copper-based fungicides for fungal issues
5. Consult your local agricultural extension officer for specific diagnosis."""
        
        elif any(word in message_lower for word in ["fertilizer", "npk", "nutrient"]):
            return """I'm currently unable to connect to my AI service. For fertilizer recommendations:
1. Get your soil tested to know current nutrient levels
2. Apply balanced NPK fertilizer based on crop requirements
3. Consider organic options like compost and vermicompost
4. Split applications are more effective than single heavy doses
5. Consult local agricultural experts for specific recommendations."""
        
        elif any(word in message_lower for word in ["crop", "grow", "plant"]):
            return """I'm currently unable to connect to my AI service. For crop selection:
1. Consider your local climate and soil type
2. Check water availability and irrigation facilities
3. Consider market demand and prices
4. Choose varieties suited to your region
5. Consult local agricultural officers for best recommendations."""
        
        else:
            return """I'm currently unable to connect to my AI service. Please try again later or consult your local agricultural extension officer for immediate assistance. You can also explore other features of Kalpataru like disease detection, crop recommendation, and fertilizer advice through the dedicated tools."""
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        return {"success": True, "message": "Conversation history cleared"}
    
    def analyze_image(
        self,
        image_data: Union[bytes, str],
        analysis_type: str = "general",
        image_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image for agricultural insights.
        For disease analysis, runs ML model first and includes results in context.
        
        Args:
            image_data: Image bytes or base64 string
            analysis_type: Type of analysis (general, disease, soil, pest)
            image_url: Optional image URL
            
        Returns:
            Analysis results with ML prediction data if applicable
        """
        analysis_prompts = {
            "general": "Analyze this agricultural image. Identify any plants, diseases, pests, or soil conditions visible. Provide helpful insights for farmers.",
            "disease": "Based on the ML model analysis provided in the context, please provide detailed advice and recommendations for this plant image. Include treatment options, preventive measures, and when to consult a local agricultural expert.",
            "soil": "Analyze this soil image. Identify the soil type, assess its apparent health, and suggest improvements for better crop production.",
            "pest": "Examine this image for any pest insects or pest damage. Identify the pest if visible and suggest control measures.",
            "crop": "Identify the crop in this image. Assess its health status and provide recommendations for better yield."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        ml_context = None
        ml_result = None
        
        # For disease or general analysis, run ML model first
        if analysis_type in ["disease", "general"]:
            # Convert image_data to bytes if needed
            if isinstance(image_data, str):
                if image_data.startswith('data:'):
                    # Extract base64 data from data URL
                    import base64
                    try:
                        base64_data = image_data.split(',', 1)[1]
                        image_bytes = base64.b64decode(base64_data)
                    except Exception as e:
                        logger.warning(f"Could not decode base64 image: {e}")
                        image_bytes = None
                else:
                    image_bytes = None
            else:
                image_bytes = image_data
            
            # Run ML disease model
            if image_bytes:
                ml_result = self._run_disease_model(image_bytes)
                if ml_result:
                    ml_context = self.build_disease_context(ml_result)
        
        # Call chat with ML context
        result = self.chat(
            message=prompt,
            image_data=image_data,
            image_url=image_url,
            include_context=True,
            ml_context=ml_context
        )
        
        # Include ML results in response if available
        if ml_result:
            result["ml_prediction"] = {
                "disease": ml_result.get("disease"),
                "confidence": ml_result.get("confidence"),
                "is_healthy": ml_result.get("is_healthy"),
                "severity": ml_result.get("severity"),
                "treatment": ml_result.get("treatment"),
                "prevention": ml_result.get("prevention")
            }
        
        return result


# Create singleton instance
ai_assistant = AIAssistant()


def get_ai_assistant() -> AIAssistant:
    """Get the AI assistant instance."""
    return ai_assistant


def chat_with_ai(
    message: str,
    image_data: Optional[Union[bytes, str]] = None,
    image_url: Optional[str] = None,
    ml_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to chat with AI assistant.
    
    Args:
        message: User's message
        image_data: Optional image data
        image_url: Optional image URL
        ml_context: Optional ML model prediction context
        
    Returns:
        Response dictionary
    """
    return ai_assistant.chat(message, image_data, image_url, ml_context=ml_context)


def analyze_agricultural_image(
    image_data: Union[bytes, str],
    analysis_type: str = "general"
) -> Dict[str, Any]:
    """
    Convenience function to analyze agricultural images.
    For disease analysis, runs ML model first and includes results.
    
    Args:
        image_data: Image data
        analysis_type: Type of analysis (general, disease, soil, pest, crop)
        
    Returns:
        Analysis results with ML prediction data if applicable
    """
    return ai_assistant.analyze_image(image_data, analysis_type)


def clear_conversation() -> Dict[str, Any]:
    """Clear the conversation history."""
    return ai_assistant.clear_history()
