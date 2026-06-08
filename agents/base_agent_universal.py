import os
import requests
import pandas as pd


class ExpertAgent:
    def __init__(self, rules, dataset_path, model_override=None):
        self.rules = rules
        self.df = pd.read_csv(dataset_path)
        self.backend, self.model, self.client = self._detect(model_override)
        print(f"[AgentIA] Backend: {self.backend} | Modelo: {self.model}")

    def _detect(self, override):
        # 1. GROQ
        if os.getenv("GROQ_API_KEY"):
            from groq import Groq
            model = override or "llama-3.1-8b-instant"
            return "groq", model, Groq()

        # 2. GOOGLE GEMINI
        if os.getenv("GEMINI_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            model = override or "gemini-1.5-flash"
            return "gemini", model, genai.GenerativeModel(model)

        # 3. HUGGINGFACE
        if os.getenv("HF_TOKEN"):
            from huggingface_hub import InferenceClient
            model = override or "HuggingFaceH4/zephyr-7b-beta"
            return "hf", model, InferenceClient(
                model=model,
                token=os.environ["HF_TOKEN"]
            )

        raise RuntimeError(
            "No se encontró ningún proveedor de IA.\n"
            "Configura una de estas variables en Streamlit Secrets:\n"
            "  GROQ_API_KEY   -> https://console.groq.com\n"
            "  GEMINI_API_KEY -> https://aistudio.google.com\n"
            "  HF_TOKEN       -> https://huggingface.co/settings/tokens"
        )

    def forward_chain(self, hechos):
        conclusiones, fired = [], []
        for r in self.rules:
            try:
                if r["cond"](hechos):
                    conclusiones.append(r["conclusion"])
                    fired.append(r["id"])
            except Exception:
                pass
        return conclusiones, fired

    def query_agent(self, hechos, conclusiones, system_prompt):
        user_msg = f"Hechos: {hechos}\nConclusiones: {conclusiones}"

        if self.backend == "groq":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg}
                ],
                max_tokens=800,
                temperature=0.3
            )
            return resp.choices[0].message.content

        elif self.backend == "gemini":
            prompt = (
                f"{system_prompt}\n\n"
                f"{user_msg}\n"
                f"Responde en español."
            )
            return self.client.generate_content(prompt).text

        elif self.backend == "hf":
            out = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg}
                ],
                max_tokens=800,
                temperature=0.3
            )
            return out.choices[0].message.content
