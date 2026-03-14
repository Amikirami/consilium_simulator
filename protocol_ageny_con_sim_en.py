import uuid
import httpx
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import json
from prompts.prompts_en import PromptsEN


model_list = ["openai/gpt-4.1-mini",
              #"microsoft/mai-ds-r1",  # Server error '503 Service Unavailable'
              "xai/grok-3-mini",
              "meta/meta-llama-3.1-405b-instruct",
              "mistral-ai/mistral-medium-2505",
              ]


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
            "max_tokens": 500,
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
                    conversation_id=conv_id
                )
                await self._deliver(msg)
                await asyncio.sleep(0.5)

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
                "conversation_id": msg.conversation_id
            })

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Full conversation saved to {filename}")

    def get_full_conversation(self, conv_id: str) -> List[ACLMessage]:
        return self.conversations.get(conv_id, [])


async def main():
    API_KEY = os.environ['GITHUB_TOKEN']

    cardio_model = GitHubModel(model_list[0], API_KEY)
    nephro_model = GitHubModel(model_list[1], API_KEY)
    hema_model = GitHubModel(model_list[2], API_KEY)

    cardio = Agent("Cardiologist", "Cardiovascular system analysis", "Expert in heart attacks and thrombosis",
                   cardio_model)
    nephro = Agent("Nephrologist", "Kidney function assessment", "Kidney failure specialist", nephro_model)
    hema = Agent("Hematologist", "Blood and clotting analysis", "Thrombosis and hematology", hema_model)

    manager = ConversationManager()
    manager.register_agent(cardio)
    manager.register_agent(nephro)
    manager.register_agent(hema)

    topic = "Patient with acute aortic thrombosis and kidney infarction"
    # facts = [
    #     "Patient 58 years old, type 2 diabetes, smoker",
    #     "Acute abdominal aortic thrombosis confirmed by CT",
    #     "Elevated CPK-MB 250 U/L, LDH 800 U/L",
    #     "Segmental infarction of right kidney (CT)",
    #     "D-dimers 4500 ng/ml"
    # ]
    facts = [PromptsEN.DIAGNOSIS, PromptsEN.LAB_RESULTS]

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


if __name__ == "__main__":
    asyncio.run(main())
