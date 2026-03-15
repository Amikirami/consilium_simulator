from dataclasses import dataclass, field
from typing import List, Dict
# from githubkit import GitHub
from prompts.prompts_en import PromptsEN
import os
import requests
import httpx
import dotenv

dotenv.load_dotenv()

MODERATOR_PROMPT = PromptsEN.MODERATOR
PATHOLOGIST_PROMPT = PromptsEN.PATHOLOGIST
SURGICAL_ONCOLOGIST_PROMPT = PromptsEN.SURGICAL_ONCOLOGIST



# TODO: hard-coded model, any other will work?
#
def get_github_models():
    url = "https://models.github.ai/catalog/models"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        
        models = response.json()
        
        print(f"{'ID':<30} | {'Publisher':<15} | {'Modalities'}")
        print("-" * 65)
        
        for model in models:
            m_id = model.get('id', 'N/A')
            publisher = model.get('publisher', 'N/A')
            modalities = ", ".join(model.get('modalities', []))
            print(f"{m_id:<30} | {publisher:<15} | {modalities}")
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")



def call_llm(system_prompt: str, messages: List[Dict[str, str]]):
    """
    Calls the GitHub Models endpoint using httpx.
    Mirrors the behavior of the working curl command.
    """

    url = "https://models.github.ai/inference/chat/completions"
    
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }


    payload = {
        # "model": "meta-llama/llama-3.1-405b-instruct",
        "model": "openai/gpt-4.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages
        ],
        "max_tokens": 800,
        "temperature": 0.2,
    }

    # httpx behaves more like curl than requests
    with httpx.Client(timeout=30.0, verify=False) as client:
        response = client.post(url, headers=headers, json=payload)

    # Raise if server returned an error
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


@dataclass
class Agent:
    name: str
    system_prompt: str
    history: List[Dict[str, str]] = field(default_factory=list)

    def send(self, content: str) -> str:
        """Send a message to this agent and get its reply."""
        self.history.append({"role": "user", "content": content})
        reply = call_llm(self.system_prompt, self.history)
        self.history.append({"role": "assistant", "content": reply})
        return reply


def run_tumor_board(case_description: str, rounds: int = 2):
    moderator = Agent("moderator", MODERATOR_PROMPT)
    pathologist = Agent("pathologist", PATHOLOGIST_PROMPT)
    surgeon = Agent("surgical_oncologist", SURGICAL_ONCOLOGIST_PROMPT)

    get_github_models()

    # 1. Moderator opens the meeting with the case
    moderator_intro_prompt = (
        f"Here is the patient case:\n\n{case_description}\n\n"
        "Please briefly summarize the case and invite the pathologist to comment first."
    )
    mod_reply = moderator.send(moderator_intro_prompt)
    print(f"\n[MODERATOR]\n{mod_reply}\n")

    for i in range(rounds):
        # 2. Pathologist responds
        path_prompt = (
            "The moderator has requested your input on this case. "
            "Please provide your pathology-focused assessment."
        )
        path_reply = pathologist.send(path_prompt)
        print(f"[PATHOLOGIST]\n{path_reply}\n")

        # 3. Moderator reacts to pathologist and invites surgeon
        mod_to_surgeon_prompt = (
            "The pathologist has just provided the above assessment. "
            "Summarize the key pathology points in 2–3 sentences and invite "
            "the surgical oncologist to comment on operability and surgical options."
        )
        mod_reply = moderator.send(mod_to_surgeon_prompt + "\n\nPathologist said:\n" + path_reply)
        print(f"[MODERATOR]\n{mod_reply}\n")

        # 4. Surgeon responds
        surg_prompt = (
            "The moderator has requested your input based on the case and the pathology summary. "
            "Please provide your surgical oncology perspective."
        )
        surg_reply = surgeon.send(surg_prompt)
        print(f"[SURGICAL ONCOLOGIST]\n{surg_reply}\n")

        # 5. Moderator synthesizes and optionally asks for further clarification
        mod_synthesis_prompt = (
            "You have heard from both the pathologist and the surgical oncologist. "
            "Briefly synthesize the discussion, highlight any uncertainties, and suggest "
            "the next steps (e.g., further tests, additional imaging, or follow-up discussion)."
        )
        mod_reply = moderator.send(
            mod_synthesis_prompt
            + "\n\nPathologist said:\n"
            + path_reply
            + "\n\nSurgical oncologist said:\n"
            + surg_reply
        )
        print(f"[MODERATOR]\n{mod_reply}\n")

    print("Tumor board simulation finished.")


if __name__ == "__main__":
    case = """
    29-year-old female, Polish descent.
    Suspicious breast lesion on ultrasound (BI-RADS 5).
    Core needle biopsy performed; preliminary report suggests invasive carcinoma.
    No known family history of breast cancer. No major comorbidities.
    Awaiting final histopathology and surgical planning.
    """
    run_tumor_board(case_description=case, rounds=1)

