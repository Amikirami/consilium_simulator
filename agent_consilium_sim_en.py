from dataclasses import dataclass, field
from typing import List, Dict
from prompts.prompts_en import PromptsEN
import os
import httpx
import logging


# Logger config
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


logging.basicConfig(
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='agconsim.log'
)
logger = logging.getLogger(__name__)


MODERATOR_PROMPT = PromptsEN.MODERATOR
PATHOLOGIST_PROMPT = PromptsEN.PATHOLOGIST
SURGICAL_ONCOLOGIST_PROMPT = PromptsEN.SURGICAL_ONCOLOGIST

model_list = ["openai/gpt-4.1-mini",
              #"microsoft/mai-ds-r1",  # Server error '503 Service Unavailable'
              "xai/grok-3-mini",
              "meta/meta-llama-3.1-405b-instruct",
              "mistral-ai/mistral-medium-2505",
              ]

#
def call_llm(model: str, system_prompt: str, messages: List[Dict[str, str]]):
    """
    Calls the GitHub Models endpoint using httpx.
    Mirrors the behavior of the working curl command.
    """
    logger.info(f"Calling LLM: model={model}, messages_count={len(messages)}")

    url = "https://models.github.ai/inference/chat/completions"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    }

    logger.debug(
        f"Sending to {url}, first message preview: {messages[0]['content'][:100]}..." if messages else "No messages")

    payload = {
        "model": f"{model}",
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages
        ],
        "max_tokens": 800,
        "temperature": 0.2,
    }

    try:
        with httpx.Client(timeout=30.0, verify=False) as client:
            logger.debug("Sending HTTP POST request...")
            response = client.post(url, headers=headers, json=payload)

        # Raise if server returned an error
        response.raise_for_status()

        result = response.json()["choices"][0]["message"]["content"]
        logger.info(f"LLM response received successfully ({len(result)} chars)")
        return result

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in call_llm: {str(e)}")
        raise


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str
    history: List[Dict[str, str]] = field(default_factory=list)

    def send(self, content: str) -> str:
        """Send a message to this agent and get its reply."""
        logger.info(f"Agent '{self.name}' received message: {content[:100]}...")

        self.history.append({"role": "user", "content": content})
        logger.debug(f"Agent history length: {len(self.history)}")

        try:
            reply = call_llm(self.model, self.system_prompt, self.history)
            self.history.append({"role": "assistant", "content": reply})
            logger.info(f"Agent '{self.name}' replied ({len(reply)} chars)")
            return reply
        except Exception as e:
            logger.error(f"Agent '{self.name}' failed to respond: {str(e)}")
            raise


def run_tumor_board(case_description: str, rounds: int = 2):

    moderator = Agent("moderator", MODERATOR_PROMPT, model_list[0])
    pathologist = Agent("pathologist", PATHOLOGIST_PROMPT, model_list[1])
    surgeon = Agent("surgical_oncologist", SURGICAL_ONCOLOGIST_PROMPT, model_list[2])

    conversation = []
    # 1. Moderator opens the meeting with the case
    moderator_intro_prompt = (
        f"Here is the patient case:\n\n{case_description}\n\n"
        "Please briefly summarize the case and invite the pathologist to comment first."
    )
    mod_reply = moderator.send(moderator_intro_prompt)
    print(f"\n[MODERATOR]\n{mod_reply}\n")
    conversation.append(f"\n[MODERATOR]\n{mod_reply}\n")

    for i in range(rounds):
        # 2. Pathologist responds
        pathologist_prompt = (
            "The moderator has requested your input on this case. "
            "Please provide your pathology-focused assessment."
        )
        pathologist_reply = pathologist.send(pathologist_prompt)
        print(f"[PATHOLOGIST]\n{pathologist_reply}\n")
        conversation.append(f"[PATHOLOGIST]\n{pathologist_reply}\n")

        # 3. Moderator reacts to pathologist and invites surgeon
        mod_to_surgeon_prompt = (
            "The pathologist has just provided the above assessment. "
            "Summarize the key pathology points in 2–3 sentences and invite "
            "the surgical oncologist to comment on operability and surgical options."
        )
        mod_reply = moderator.send(mod_to_surgeon_prompt + "\n\nPathologist said:\n" + pathologist_reply)
        print(f"[MODERATOR]\n{mod_reply}\n")
        conversation.append(f"[MODERATOR]\n{mod_reply}\n")

        # 4. Surgeon responds
        surg_prompt = (
            "The moderator has requested your input based on the case and the pathology summary. "
            "Please provide your surgical oncology perspective."
        )
        surg_reply = surgeon.send(surg_prompt)
        print(f"[SURGICAL ONCOLOGIST]\n{surg_reply}\n")
        conversation.append(f"[SURGICAL ONCOLOGIST]\n{surg_reply}\n")

        # 5. Moderator synthesizes and optionally asks for further clarification
        mod_synthesis_prompt = (
            "You have heard from both the pathologist and the surgical oncologist. "
            "Briefly synthesize the discussion, highlight any uncertainties, and suggest "
            "the next steps (e.g., further tests, additional imaging, or follow-up discussion)."
        )
        mod_reply = moderator.send(
            mod_synthesis_prompt
            + "\n\nPathologist said:\n"
            + pathologist_reply
            + "\n\nSurgical oncologist said:\n"
            + surg_reply
        )
        print(f"[MODERATOR]\n{mod_reply}\n")
        conversation.append(f"[MODERATOR]\n{mod_reply}\n")

    print("Tumor board simulation finished.")

    # Write conversation
    with open("converstaion.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(conversation))


if __name__ == "__main__":
    case = PromptsEN.DIAGNOSIS + PromptsEN.LAB_RESULTS
    run_tumor_board(case_description=case, rounds=1)

