import os
import yaml
import requests
import re

class LLMReport:
    def __init__(self, model_url: str = None, model_name: str = None, report_path: str = None):
        # Use environment variables or fallback defaults
        self.ollama_base_url = (model_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.report_path = report_path or os.getenv("REPORT_PATH", "../report/final_report.txt")

    def generate_report(self):
        with open("../prompts/prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)

        template = prompts["compliance_analysis"]["template"]

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

        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()

        result = response.json()["message"]["content"]

        match = re.search(r"(The policy is .*?)$", result, re.DOTALL | re.IGNORECASE)

        if match:
            result = match.group(1)

        with open("../report/llm_report.txt", "w") as f:
            f.write(result)

        return result


if __name__ == "__main__":
    llm_report = LLMReport()  # Automatically pulls from env or defaults
    llm_report.generate_report()
