import yaml
import requests

class LLMReport:
    def __init__(self, model_url: str, model_name: str, report_path: str):
        self.ollama_base_url = model_url.rstrip("/")
        self.model_name = model_name
        self.report_path = report_path

    def generate_report(self):
        with open("./prompts/prompts.yaml", "r") as f:
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
        print("\n\nLLM Compliance Report:\n")
        print(result)

        with open("report/llm_report.txt", "w") as f:
            f.write(result)


if __name__ == "__main__":
    ollama_base_url = "http://localhost:11434"
    model_name = "llama3.2:3b"
    report_path = "./report/final_report.txt"

    llm_report = LLMReport(ollama_base_url, model_name, report_path)
    llm_report.generate_report()
