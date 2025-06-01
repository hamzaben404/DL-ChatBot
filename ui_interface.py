# ui_interface.py
"""
Safe interface between Gradio UI and RAG system
Uses process isolation to prevent conflicts
"""
import subprocess
import json
import tempfile
import os

def run_rag_query(question: str, top_k: int = 3) -> dict:
    """Run RAG query in isolated process"""
    try:
        # Create temp input file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump({"question": question, "top_k": top_k}, f)
            input_path = f.name
        
        # Create temp output file
        output_path = tempfile.mktemp(suffix='.json')
        
        # Run command
        cmd = [
            "python", "-m", "scripts.generate_answer",
            "--input", input_path,
            "--output", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Check results
        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Command failed ({result.returncode}):\n{result.stderr}"
            }
        
        # Load output
        if os.path.exists(output_path):
            with open(output_path) as f:
                return json.load(f)
        return {
            "status": "error",
            "message": "Output file not created"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }
    
    finally:
        # Cleanup temp files
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                os.unlink(path)