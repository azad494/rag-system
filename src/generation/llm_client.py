import sys
from typing import List, Dict, Any

try:
    import ollama
except ImportError:
    print("\n❌ Error: 'ollama' library missing. Run: pip install ollama")
    sys.exit(1)

class ClinicalGenerationEngine:
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name

    def generate_clinical_answer_stream(self, question: str, context_blocks: List[Dict[str, Any]]):
        """
        Yields live word tokens directly from the model layers.
        Enforces 100% string-isolated token generation.
        """
        if not context_blocks:
            yield "❌ Refusal Guard: The requested query cannot be verified against the provided clinical data logs. Information unavailable."
            return

        # Keep context highly dense to save CPU calculation cycles
        formatted_context = ""
        for chunk in context_blocks:
            formatted_context += f"[Postgres Chunk ID: {chunk['id']}]: {chunk['page_content']}\n"

        system_instruction = (
            "You are a strict Clinical Support Decision AI. Answer the question using ONLY the context data listed below.\n"
            "Constraints:\n"
            "1. Do not use outside facts. For every claim you make, explicitly append the Postgres Chunk ID at the end of the sentence (e.g., [ID: 9]).\n"
            "2. If the context does not explicitly contain the exact answer facts, state exactly: 'I cannot answer this based on the available clinical logs.'\n"
            "3. Keep your output brief, clean, and direct to the point."
        )

        user_prompt = f"VERIFIED CONTEXT VAULT:\n{formatted_context}\n\nUSER QUESTION: {question}\n\nCLINICAL SYNTHESIS:"

        try:
            # Yield chunks natively
            response_stream = ollama.generate(
                model=self.model_name,
                system=system_instruction,
                prompt=user_prompt,
                stream=True,  # Activates the streaming event channel port
                options={
                    "temperature": 0.0,
                    "num_predict": 200,   # Prevent infinite semantic looping
                    "num_ctx": 2048       # Keep memory footprint bounded
                }
            )
            
            for chunk in response_stream:
                # Extract the underlying primitive string value securely!
                token_text = chunk.get("response", "")
                if token_text:
                    yield token_text
                
        except Exception as e:
            yield f"❌ Critical failure connecting to local Ollama server container: {str(e)}"