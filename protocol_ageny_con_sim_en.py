import random
import uuid
from traceback import print_tb

import httpx
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import json
import requests
from prompts.prompts_en import PromptsEN
import re
from time import sleep


# TO-DO: Zabezpieczenie gdy jest 403 przy próbie użycia modelu
# TODO: Or not TO-DO: 429: Too many requests
# TODO: Handle 400: {"error":{"code":"unavailable_model","message":"Unavailable model: gpt-5"


def get_github_models():
    url = "https://models.github.ai/catalog/models"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    try:
        model_ids = []
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors

        models = response.json()

        print(f"{'ID':<44} | {'Publisher':<15} | {'Modalities'}")
        print("-" * 65)

        for model in models:
            m_id = model.get('id', 'N/A')
            publisher = model.get('publisher', 'N/A')
            modalities = ", ".join(model.get('supported_input_modalities', []))
            print(f"{m_id:<44} | {publisher:<15} | {modalities}")
            model_ids.append(m_id)
        return model_ids
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")


@dataclass
class TreatmentProposal:
    diagnoses_confirmed: List[str]
    treatment_plan: List[str]
    risks: List[str]
    next_steps: List[str]
    source_specialists: List[str]
    notes: Optional[str] = None

# diagnoses_confirmed: List[str]
# diagnoses_suspected: List[str]          # optional but very useful
# treatment_plan: List[str]
# next_steps: List[str]
# risks: List[str]                        # extracted clinical risks
# notes: Optional[str] = None
# source_specialists: List[str]           # who contributed

@dataclass
class ACLMessage:
    performative: str
    sender: str
    receiver: str
    content: str
    conversation_id: str
    metadata: Optional[Dict] = None


class GitHubModel:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://models.github.ai/inference/chat/completions"

        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.api_key}",
            "X-GitHub-Api-Version": "2026-03-10",
            "Content-Type": "application/json"
        }

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.2,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.url, headers=self.headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                return "No model response"

        except httpx.HTTPStatusError as e:
            print(f"API error {e.response.status_code}: {e.response.text[:100]}")
            return f"API error {e.response.status_code}: {e.response.text[:100]}"
        except Exception as e:
            return f"Error: {str(e)}"

    async def generate_role(self, messages) -> str:
        """
        messages: list of dicts, each dict = {"role": "...", "content": "..."}
        """

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.2,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(self.url, headers=self.headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                return "No model response"

        except httpx.HTTPStatusError as e:
            print(f"API error {e.response.status_code}: {e.response.text[:100]}")
            return f"API error {e.response.status_code}: {e.response.text[:100]}"
        except Exception as e:
            return f"Error: {str(e)}"


class Agent:
    def __init__(self, name: str, role: str, specialization: str, model: GitHubModel):
        self.name = name
        self.role = role
        self.specialization = specialization
        self.model = model
        self.memory: List[str] = []

    def get_my_model(self) -> str:
        return self.model.model_name

    def add_fact(self, fact: str):
        self.memory.append(fact[:200])

    def _build_context(self, incoming: str) -> str:
        recent_memory = "\n".join(self.memory[-3:])
        return f"""You are {self.name} ({self.role}).
            Specialization: {self.specialization}
            
            MEMORY ({len(self.memory)} facts):
            {recent_memory}
            
            NEW MESSAGE:
            {incoming}
            
            RESPOND according to your role, use facts from memory. Be brief and specific."""

    async def handle_message(self, msg: ACLMessage) -> Optional[ACLMessage]:
        prompt = self._build_context(msg.content)
        await asyncio.sleep(10)
        reply_text = await self.model.generate(prompt)

        performative_map = {
            "inform": "inform",
            "request": "propose",
            "propose": "agree",
            "agree": "inform",
            "reject": "inform"
        }
        next_performative = performative_map.get(msg.performative, "inform")

        if len(reply_text) > 20 and "Error" not in reply_text:
            self.add_fact(reply_text)

        return ACLMessage(
            performative=next_performative,
            sender=self.name,
            receiver=msg.sender,
            content=reply_text,
            conversation_id=msg.conversation_id
        )


class ConversationManager:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.conversations: Dict[str, List[ACLMessage]] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}

    def register_agent(self, agent: Agent):
        self.agents[agent.name] = agent
        print(f"Agent registered: {agent.name}")

    async def start_conversation(self, topic: str, facts: List[str], participants: List[str]) -> str:
        conv_id = str(uuid.uuid4())
        self.conversations[conv_id] = []

        print(f"Starting conversation: {topic}")

        for name in participants:
            if name in self.agents:
                for fact in facts:
                    self.agents[name].add_fact(fact)

        topic_msg = f"TOPIC: {topic}\n\nFACTS:\n" + "\n".join(facts)
        for name in participants:
            if name in self.agents:
                msg = ACLMessage(
                    performative="inform",
                    sender="System",
                    receiver=name,
                    content=topic_msg,
                    conversation_id=conv_id,
                    metadata={"receiver_model": self.agents[name].get_my_model()}
                )
                await self._deliver(msg)
                await asyncio.sleep(1)

        return conv_id

    async def _deliver(self, msg: ACLMessage):
        self.conversations[msg.conversation_id].append(msg)

        if msg.receiver in self.agents:
            agent = self.agents[msg.receiver]
            reply = await agent.handle_message(msg)
            if reply:
                self.conversations[msg.conversation_id].append(reply)
                print(f"[{msg.sender}->{reply.receiver}] {reply.performative}: {reply.content[:80]}...")

    async def broadcast_request(self, conv_id: str, sender: str, content: str, recipients: List[str]):
        tasks = []
        for recipient in recipients:
            if recipient in self.agents:
                msg = ACLMessage(
                    performative="request",
                    sender=sender,
                    receiver=recipient,
                    content=content,
                    conversation_id=conv_id
                )
                tasks.append(self._deliver(msg))

        await asyncio.gather(*tasks, return_exceptions=True)

    def print_conversation(self, conv_id: str):
        conversation = self.conversations.get(conv_id, [])
        print(f"\n{'=' * 60}")
        print(f"CONVERSATION {conv_id}")
        print(f"Total messages: {len(conversation)}")
        print(f"{'=' * 60}")
        for i, msg in enumerate(conversation):
            print(f"[{i:2d}] {msg.sender} -> {msg.receiver} ({msg.performative})")
            print(f"    {msg.content}")
            print("-" * 60)

    def save_conversation(self, conv_id: str, filename: str = None):
        conversation = self.conversations.get(conv_id, [])
        if not filename:
            filename = f"conversation_{conv_id[:8]}.json"

        data = []
        for i, msg in enumerate(conversation):
            data.append({
                "id": i,
                "sender": msg.sender,
                "receiver": msg.receiver,
                "performative": msg.performative,
                "content": msg.content,
                "conversation_id": msg.conversation_id,
                "metadata": msg.metadata
            })

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Full conversation saved to {filename}")

    def get_full_conversation(self, conv_id: str) -> List[ACLMessage]:
        return self.conversations.get(conv_id, [])


def asses_result(conv_summary:TreatmentProposal, reference:TreatmentProposal):
    """Compares the discussion result to the reference data"""
    pass


def merge_proposals(reports: list[ACLMessage]) -> str:
    """
    Takes a list of ACLMessage dataclass objects and returns a single
    merged text containing only those with performative == 'propose'.
    """
    parts = []

    for report in reports:
        if report.performative == "propose":
            parts.append(f"{report.sender}:\n{report.content}")

    return "\n\n".join(parts)


async def main():
    API_KEY = os.environ['GITHUB_TOKEN']
    K = 3

    avail_models = get_github_models()
    model_list = random.sample(avail_models, K)
    print(f"Selected models: {model_list}")

    cardio_model = GitHubModel(model_list[0], API_KEY)
    sleep(1)
    nephro_model = GitHubModel(model_list[1], API_KEY)
    sleep(1)
    hema_model = GitHubModel(model_list[2], API_KEY)
    sleep(1)

    cardio = Agent("Cardiologist", "Cardiovascular system analysis", "Expert in heart attacks and thrombosis",
                   cardio_model)
    nephro = Agent("Nephrologist", "Kidney function assessment", "Kidney failure specialist", nephro_model)
    hema = Agent("Hematologist", "Blood and clotting analysis", "Thrombosis and hematology", hema_model)

    manager = ConversationManager()
    manager.register_agent(cardio)
    manager.register_agent(nephro)
    manager.register_agent(hema)

    topic = "A clinical assessment is required for a patient presenting with the following symptoms. "

    # Rozdzielenie "facts" na jednoliniową listę - do przechowania w pamieci agenta
    facts = re.split(r"\n+\s*", PromptsEN.DIAGNOSIS)
    facts.extend(re.split(r"\n+\s*", PromptsEN.LAB_RESULTS))

    conv_id = await manager.start_conversation(topic, facts, ["Cardiologist", "Nephrologist", "Hematologist"])
    await asyncio.sleep(2)

    await manager.broadcast_request(
        conv_id,
        "Attending Physician",
        "Please provide urgent risk assessment and treatment plan. Time critical!",
        ["Cardiologist", "Nephrologist", "Hematologist"]
    )

    await asyncio.sleep(3)

    # TERAZ CALA KONWERSACJA
    manager.print_conversation(conv_id)
    manager.save_conversation(conv_id)

    # Lub pobierz liste obiektow
    full_history = manager.get_full_conversation(conv_id)
    print(f"\nLiczba wiadomosci: {len(full_history)}")

    # Podsumowanie dyskusji
    specialist_replies = merge_proposals(full_history)
    print("\n####"*4, "\nProposals:", specialist_replies)
    user_content = PromptsEN.SUMMARIZER_USER.format(INPUT_DATA=specialist_replies)
    print("\n####" * 4, "\nUser propmt content:", specialist_replies)
    #
    s_name = "openai/gpt‑4o"
    summarizer = GitHubModel(s_name, API_KEY)
    summarizer_prompts = [
        {"role": "system", "content": PromptsEN.SUMMARIZER_SYSTEM},
        {"role": "user", "content": user_content},
    ]


    conv_summary = await summarizer.generate_role(summarizer_prompts)
    print("\n####"*4, "\nConv summary:", conv_summary)

    #
    # reference_summary = ""
    # asses_result(conv_summary, reference_summary)


if __name__ == "__main__":
    # get_github_models()
    asyncio.run(main())
