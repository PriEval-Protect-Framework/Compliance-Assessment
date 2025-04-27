import yaml
import requests
import re
import os

class LLMReport:
    def __init__(self, model_url: str, model_name: str, report_path: str):
        print(f"[LLMReport] Initializing with model_url={model_url}, model_name={model_name}, report_path={report_path}")
        self.ollama_base_url = model_url.rstrip("/")
        self.model_name = model_name
        self.report_path = report_path

    def generate_report(self):
        print("[LLMReport] Starting report generation")
        try:
            prompts_path = "../prompts/prompts.yaml"
            print(f"[LLMReport] Loading prompts from {prompts_path}")
            
            if not os.path.exists(prompts_path):
                print(f"[LLMReport] ERROR: Prompts file not found at {os.path.abspath(prompts_path)}")
                return "Error: Prompts file not found"
                
            with open(prompts_path, "r") as f:
                prompts = yaml.safe_load(f)
                print(f"[LLMReport] Loaded prompts: {list(prompts.keys())}")

            if "compliance_analysis" not in prompts or "template" not in prompts["compliance_analysis"]:
                print("[LLMReport] ERROR: Missing required 'compliance_analysis.template' in prompts file")
                return "Error: Invalid prompts file structure"

            template = prompts["compliance_analysis"]["template"]
            print(f"[LLMReport] Template loaded, length: {len(template)} characters")

            if not os.path.exists(self.report_path):
                print(f"[LLMReport] ERROR: Report file not found at {os.path.abspath(self.report_path)}")
                return "Error: Report file not found"

            with open(self.report_path, "r") as f:
                report_text = f.read()
            print(f"[LLMReport] Report loaded, length: {len(report_text)} characters")

            final_prompt = template.format(report=report_text)
            print(f"[LLMReport] Final prompt prepared, length: {len(final_prompt)} characters")

            ollama_url = f"{self.ollama_base_url}/api/chat"
            print(f"[LLMReport] Sending request to Ollama at {ollama_url}")
            
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
                print(f"[LLMReport] Making request to Ollama with model {self.model_name}")
                response = requests.post(ollama_url, json=payload, timeout=120)
                print(f"[LLMReport] Response status code: {response.status_code}")
                response.raise_for_status()
                print("[LLMReport] Response received successfully")
            except requests.exceptions.RequestException as e:
                print(f"[LLMReport] ERROR: Request to Ollama API failed: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"[LLMReport] Response status: {e.response.status_code}")
                    print(f"[LLMReport] Response body: {e.response.text[:500]}...")
                import traceback
                traceback.print_exc()
                return f"Error connecting to Ollama: {str(e)}"

            try:
                print("[LLMReport] Parsing JSON response")
                response_json = response.json()
                print("[LLMReport] JSON parsed successfully")
                
                result = response_json["message"]["content"]
                print(f"[LLMReport] Result extracted, length: {len(result)} characters")
            except (KeyError, ValueError) as e:
                print(f"[LLMReport] ERROR: Failed to parse response: {e}")
                print(f"[LLMReport] Raw response: {response.text[:500]}...")
                import traceback
                traceback.print_exc()
                return f"Error parsing Ollama response: {str(e)}"

            print("[LLMReport] Extracting final result using regex")
            match = re.search(r"(The policy is .*?)$", result, re.DOTALL | re.IGNORECASE)

            if match:
                print("[LLMReport] Pattern matched, extracting")
                result = match.group(1)
            else:
                print("[LLMReport] Pattern not matched, using full result")

            output_path = "../report/llm_report.txt"
            print(f"[LLMReport] Writing result to {output_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print("[LLMReport] Result written successfully")

            return result
            
        except Exception as e:
            print(f"[LLMReport] CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating report: {str(e)}"