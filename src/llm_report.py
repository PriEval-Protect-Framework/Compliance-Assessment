import os
import yaml
import requests
import re
import os

class LLMReport:
    def __init__(self, model_url: str = None, model_name: str = None, report_path: str = None):
        # Use environment variables or fallback defaults
        self.ollama_base_url = (model_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.report_path = report_path or os.getenv("REPORT_PATH", "../report/final_report.txt")

    def generate_report(self):
        print("[LLMReport] Starting report generation")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompts_path = os.path.join(BASE_DIR, "prompts")
        REPORT_DIR = os.path.join(BASE_DIR, "report")



        
        if not os.path.exists(prompts_path):
            return "Error: Prompts file not found"
            
        with open(f"{prompts_path}/prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
        if "compliance_analysis" not in prompts or "template" not in prompts["compliance_analysis"]:
            return "Error: Invalid prompts file structure"
        template = prompts["compliance_analysis"]["template"]
        if not os.path.exists(self.report_path):
            return "Error: Report file not found"
        with open(self.report_path, "r") as f:
            report_text = f.read()
        final_prompt = template.format(report=report_text)
        ollama_url = f"{self.ollama_base_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            "stream": False
        }
        try:
            response = requests.post(ollama_url, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"[LLMReport] Error during Ollama request: {e}")
            return f"LLM report generation failed: {e}"
        response.raise_for_status()
        result = response.json()["message"]["content"]
        match = re.search(r"(The policy is .*?)$", result, re.DOTALL | re.IGNORECASE)
        if match:
            result = match.group(1)
            output_path = f"{REPORT_DIR}/llm_report.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
        return result


if __name__ == "__main__":
    llm_report = LLMReport()  
    llm_report.generate_report()
