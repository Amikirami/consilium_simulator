import random
import uuid
import httpx
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import asyncio
import json
import requests
from prompts.prompts_en import PromptsEN
import re
from time import sleep
import numpy as np
import logging
import pandas as pd
from embedding_openai import EmbeddingClient
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='out-prot/portocol_agntconsim.log'
)
logger = logging.getLogger(__name__)


@dataclass
class TreatmentProposal:
    diagnoses_confirmed: List[str]
    diagnoses_suspected: List[str]
    treatment_plan: List[str]
    risks: List[str]
    next_steps: List[str]
    # source_specialists: List[str]
    notes: Optional[str] = None


@dataclass
class ACLMessage:
    performative: str
    sender: str
    receiver: str
    content: str
    conversation_id: str
    metadata: Optional[Dict] = None


def remove_think_blocks(text: str) -> str:
    # usuwa wszystko pomiędzy <think> ... </think>, łącznie ze znacznikami
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


class OpenAIModel:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    async def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "developer",
                "content": (
                    "Do not reveal chain-of-thought. "
                    "Provide only the final answer or a short explanation. "
                    "nothink /nothink /no_think"
                )
            },
            {
                "role": "user",
                "content": (
                        "Respond concisely. Do not include <think> or hidden reasoning. /nothink "
                        + prompt
                )
            }
        ]

        logger.info(f"Calling LLM (Responses API): model={self.model_name}, user_prompt_len={len(prompt)}")
        sleep(1)

        print("genrate:model:", self.model_name)
        reasoning_params = (
            {"reasoning": {"effort": "low"}}
            if self.model_name in {"o1", "o3", "o3-mini", "o4", "gpt-5", "gpt-5-mini"}
            else {"temperature": 0.2}
        )
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                max_output_tokens=2048,
                **reasoning_params
            )

            # Responses API returns output in response.output_text
            return response.output_text.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"

    async def generate_role(self, messages) -> str:
        logger.info(f"Calling LLM (Responses API): generate_role model={self.model_name}")

        reasoning_params = (
            {"reasoning": {"effort": "low"}}
            if self.model_name in {"o1", "o3", "o3-mini", "o4", "gpt-5", "gpt-5-mini"}
            else {"temperature": 0.2}
        )
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=messages,
                max_output_tokens=2048,
                # temperature=0.2
                **reasoning_params
            )

            return response.output_text.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"


def analyze_agent_response(reply):
    reply_lower = reply.lower()

    # Priorytet 1: wyraźne deklaracje
    if re.search(r'\b(agree|accept|support|yes)\b', reply_lower):
        return "agree"
    if re.search(r'\b(reject|no|disagree|oppose)\b', reply_lower):
        return "reject"

    # Priorytet 2: pytania/propozycje
    if re.search(r'\?', reply_lower) or re.search(r'\bpropose\b', reply_lower):
        return "counter-propose"

    # Priorytet 3: informacja/fakty
    # return "inform"
    return "agree"


class Agent:
    def __init__(self, name: str, role: str, specialization: str, model: OpenAIModel):
        self.name = name
        self.role = role
        self.specialization = specialization
        self.model = model
        self.memory: List[str] = []

    def get_my_model(self) -> str:
        return self.model.model_name

    def add_fact(self, fact: str):
        print(f"{self.name} - fact len:", len(fact))
        # self.memory.append(fact[:200])
        if len(fact) > 0:
            self.memory.append(fact)

    def _build_context(self, incoming: str) -> str:
        # recent_memory = "\n".join(self.memory[-3:])
        recent_memory = "\n".join(self.memory[-8:])
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
        try:
            reply_text = await self.model.generate(prompt)
        except httpx.HTTPStatusError as e:
            raise
        sleep(10)

        # performative_map = {
        #     "inform": "inform",
        #     "request": "propose",
        #     "propose": "agree",
        #     "agree": "inform",
        #     "reject": "inform"
        # }
        # next_performative = performative_map.get(msg.performative, "inform")

        performative_map = {
            "inform": "inform",
            "request": "propose",
            "propose": ["agree", "reject", "counter-propose"],
            "agree": "inform",
            "reject": "propose",
            "counter-propose": "propose"
        }

        if msg.performative == "propose":
            next_act = analyze_agent_response(reply_text)
            next_performative = next_act
            logger.info(f"Agent {self.name} response was: {next_act}")
        else:
            next_performative = performative_map[msg.performative]

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
        print(f"Agent registered: {agent.name}: {agent.model.model_name}")

    async def start_conversation(self, topic: str, facts: List[str], participants: List[str]) -> str:
        conv_id = str(uuid.uuid4())
        logging.info(f"Starting conversation id={conv_id}")
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
                await asyncio.sleep(10)
                sleep(5)

        return conv_id

    async def _deliver(self, msg: ACLMessage):
        self.conversations[msg.conversation_id].append(msg)

        if msg.receiver in self.agents:
            agent = self.agents[msg.receiver]
            reply = await agent.handle_message(msg)
            if reply:
                self.conversations[msg.conversation_id].append(reply)
                # print(f"[{msg.sender}->{reply.receiver}] {reply.performative}: {reply.content[:80]}...")
                print(f"[{msg.sender}->{reply.receiver}] {reply.performative}: {reply.content}")

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
            filename = f"out-prot/conversation_{conv_id[:8]}.json"

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


# ----------------------------------------------
# Compare

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# async def embed_fn(text: str, token: str, model: str = "openai/text-embedding-3-large"):
#     url = "https://models.github.ai/inference/embeddings"
#     headers = {
#         "Authorization": f"Bearer {token}",
#         "X-Model": model,
#     }
#     payload = {
#         "input": text,
#         "model": model
#     }
#     sleep(10)
#     async with httpx.AsyncClient(timeout=60.0) as client:
#         resp = await client.post(url, headers=headers, json=payload)
#         resp.raise_for_status()
#         data = resp.json()
#
#     # GitHub Models zwraca embedding w data["data"][0]["embedding"]
#     print("Embeddings::", str(data)[:100], "...", str(data)[-100:])
#     return data["data"][0]["embedding"]


# !!! G..no się zrobi darmochą !!!
# async def list_embedding_similarity(list1, list2, embed_fn):
#     if not list1 and not list2:
#         return 1.0
#     if not list1 or not list2:
#         return 0.0
#
#     # embeddingi dla obu list
#     emb1 = [await embed_fn(x, os.environ['GITHUB_TOKEN']) for x in list1]
#     sleep(30)
#     emb2 = [await embed_fn(x, os.environ['GITHUB_TOKEN']) for x in list2]
#     sleep(30)
#
#     scores = []
#     for e1 in emb1:
#         best = max(cosine_similarity(e1, e2) for e2 in emb2)
#         scores.append(best)
#
#     return round(float(sum(scores) / len(scores)), 3)


# async def compare_conv_summaries_embeddings(a: TreatmentProposal, b: TreatmentProposal, embed_fn):
#     return {
#         # "diagnoses_confirmed": await list_embedding_similarity(a.diagnoses_confirmed, b.diagnoses_confirmed, embed_fn),
#         "diagnoses_suspected": await list_embedding_similarity(a.diagnoses_suspected, b.diagnoses_suspected, embed_fn),
#         "treatment_plan": await list_embedding_similarity(a.treatment_plan, b.treatment_plan, embed_fn),
#         "risks": await list_embedding_similarity(a.risks, b.risks, embed_fn),
#         "next_steps": await list_embedding_similarity(a.next_steps, b.next_steps, embed_fn),
#         # "source_specialists": await list_embedding_similarity(a.source_specialists, b.source_specialists, embed_fn),
#         "notes": await list_embedding_similarity(a.notes, b.notes, embed_fn),
#     }

#     diagnoses_confirmed: List[str]
#     diagnoses_suspected: List[str]
#     treatment_plan: List[str]
#     risks: List[str]
#     next_steps: List[str]
#     source_specialists: List[str]
#     notes: Optional[str] = None

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


from dataclasses import fields


def parse_conv_summary(text: str) -> TreatmentProposal:
    # wyciągamy JSON z tekstu
    import json, re
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Nie znaleziono JSON w tekście")

    data = json.loads(match.group(0))

    # lista dozwolonych pól
    allowed = {f.name for f in fields(TreatmentProposal)}

    # filtrujemy
    filtered = {k: v for k, v in data.items() if k in allowed}

    return TreatmentProposal(**filtered)


def append_to_json_list(path, item):
    data = []

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    if not isinstance(data, list):
        raise ValueError("Plik JSON nie zawiera listy!")

    data.append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def main():
    # API_KEY = os.environ['GITHUB_TOKEN']
    API_KEY = None
    medical_case_number = 2

    logging.info(f"Staring main, med. case: {medical_case_number}")
    print(f"Staring main, medical case: {medical_case_number}")

    def counter(N):
        i = 0
        while True:
            yield i
            i = (i + 1) % (N + 1)

    # avail_models = get_github_models()

    # Random choice ---------------------------------------------------------------------------
    # non_completion = set()
    # model_list = []
    #
    # while len(model_list) < K:
    #     candidates = [x for x in avail_models if x not in non_completion]
    #     model_ = random.choice(candidates)
    #
    #     if not await model_supports_chat(model_, API_KEY):
    #         non_completion.add(model_)
    #     else:
    #         model_list.append(model_)
    #     sleep(2)

    #
    # Specified choice
    # Check if available
    # model_list = []
    # non_completion = set()
    # candidates = [
    #     # "openai/gpt-4o",
    #     "openai/gpt-4o-mini",
    #     # "openai/gpt-5",
    #     "openai/gpt-5-mini",
    #     # "openai/gpt-5-chat",
    #     "openai/o1-preview",
    #     # "openai/o1",
    #     "openai/o1-mini",
    #     # "openai/o3",
    #     "openai/o3-mini",
    #     # "microsoft/phi-4-reasoning",  # timeout
    #     "microsoft/phi-4-mini-reasoning",
    #     "deepseek/deepseek-r1",
    #     # "deepseek/deepseek-r1-0528",
    #     "meta/meta-llama-3.1-405b-instruct",
    #     # "meta/llama-3.3-70b-instruct",
    # ]
    # while len(model_list) < K:
    #     candidates = [x for x in candidates if x not in non_completion]
    #     if not candidates:
    #         break
    #     model_ = candidates[0]
    #     print(f"Checking model: {model_}")
    #     if not await model_supports_chat(model_, API_KEY):
    #         non_completion.add(model_)
    #     else:
    #         model_list.append(model_)
    #         non_completion.add(model_)
    #     sleep(10)
    #
    # num_models = len(model_list)
    # c = counter(num_models-1)

    model_list = [
        "gpt-4o",
        "gpt-5-mini",
        "o1",
        "o3",
        "o3-mini",
        "o4‑mini‑deep‑research",
        "gpt-5.1",
    ]
    num_models = len(model_list)
    c = counter(num_models - 1)

    print(f"Selected models: {model_list}\n")
    # print("openai/gpt-4o-mini::", await model_supports_chat("openai/gpt-4o-mini", API_KEY))

    # Model for structurize summarize
    sum_struct_model_name = "gpt-4o-mini"
    # sum_struct_model_name = "cohere/cohere-command-r-plus-08-2024"
    # sum_struct_model_name = "xai/grok-3"
    print("Summarize structured model: ", sum_struct_model_name)
    # if not await model_supports_chat(sum_struct_model_name, API_KEY):
    #     return

    # cardio_model = GitHubModel(model_list[0], API_KEY)
    nephro_model = OpenAIModel(model_list[next(c)], API_KEY)
    hema_model = OpenAIModel(model_list[next(c)], API_KEY)
    vascular_model = OpenAIModel(model_list[next(c)], API_KEY)
    endo_model = OpenAIModel(model_list[next(c)], API_KEY)

    # cardio = Agent("Cardiologist", "Cardiovascular system analysis", "Expert in heart attacks and thrombosis",
    #                cardio_model)
    nephro = Agent("Nephrologist",
                   "Kidney function assessment",
                   "Kidney failure specialist",
                   nephro_model)
    hema = Agent("Hematologist",
                 "Blood and clotting analysis",
                 "Thrombosis and hematology",
                 hema_model)

    # vascular surgeon
    # endovascular interventionist
    vascular = Agent(
        "Vascular Surgeon",
        "Peripheral vascular and arterial intervention",
        "Expert in surgical management of arterial occlusions and limb ischemia",
        vascular_model
    )

    endo = Agent(
        "Endovascular Interventionist",
        "Minimally invasive vascular procedures",
        "Specialist in catheter‑based thrombolysis, angioplasty, and stent placement",
        endo_model
    )

    manager = ConversationManager()
    # manager.register_agent(cardio)
    manager.register_agent(nephro)
    manager.register_agent(hema)

    manager.register_agent(vascular)
    manager.register_agent(endo)

    topic = "A clinical assessment is required for a patient presenting with the following symptoms. "

    # Rozdzielenie "facts" na jednoliniową listę - do przechowania w pamieci agenta
    facts = re.split(r"\n+\s*", PromptsEN.DIAGNOSIS)
    facts.extend(re.split(r"\n+\s*", PromptsEN.LAB_RESULTS))

    # conv_id = await manager.start_conversation(topic, facts, ["Cardiologist", "Nephrologist", "Hematologist"])
    conv_id = await manager.start_conversation(topic, facts,
                                               # ["Cardiologist",
                                               ["Nephrologist", "Hematologist",
                                                "Vascular Surgeon", "Endovascular Interventionist"])
    await asyncio.sleep(2)

    await manager.broadcast_request(
        conv_id,
        "Attending Physician",
        "Please provide urgent risk assessment and treatment plan. Time critical!",
        # ["Cardiologist",
        ["Nephrologist", "Hematologist",
         "Vascular Surgeon", "Endovascular Interventionist"]
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
    print("\n####" * 4, "\nProposals:", specialist_replies)
    user_content = PromptsEN.SUMMARIZER_USER.format(INPUT_DATA=specialist_replies)
    print("\n####" * 4, "\nUser prompt content:", specialist_replies)
    #

    summarizer = OpenAIModel(sum_struct_model_name, API_KEY)
    summarizer_prompts = [
        {"role": "developer", "content": PromptsEN.SUMMARIZER_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    conv_summary = await summarizer.generate_role(summarizer_prompts)
    print("\n####" * 4, "\nConv summary:", conv_summary)

    # Append summary to file
    add_data = {
        "id": -1,
        "sender": "Summary",
        "receiver": "",
        "performative": "",
        "content": conv_summary,
        "conversation_id": "",
    }

    filename = f"out-prot/conversation_{conv_id[:8]}.json"
    # Hacky add summary to json
    with open(filename, "a+", encoding="utf-8") as f:
        f.seek(0)
        txt = f.read()
        pos = txt.rfind("]")
        if pos != -1:
            txt = txt[:pos]
        f.seek(0)
        f.write(txt)
        f.write(",\n")
        json.dump(add_data, f, indent=2, ensure_ascii=False)
        f.write("]\n")

    append_to_json_list(filename, add_data)

    print(f"Full conversation saved to {filename}")

    # add_data = ACLMessage(
    #                 performative="",
    #                 sender="Summary",
    #                 receiver="",
    #                 content=conv_summary,
    #                 conversation_id="",
    # )
    # full_history.append(add_data)

    #
    reference = PromptsEN.DIAGNOSIS + "\n" + PromptsEN.REFERENCE_TREATMENT
    summarizer_prompts = [
        {"role": "developer", "content": PromptsEN.SUMMARIZER_SYSTEM},
        {"role": "user", "content": reference},
    ]
    print("Creating reference summary...")
    sleep(10)
    reference_summary = await summarizer.generate_role(summarizer_prompts)
    cs = parse_conv_summary(conv_summary)
    rs = parse_conv_summary(reference_summary)

    print("-----------")
    print("Converstion - structured::")
    print(json.dumps(asdict(cs), indent=2, ensure_ascii=False))
    print("---------")
    print("Reference   - structured::")
    print(json.dumps(asdict(rs), indent=2, ensure_ascii=False))

    # asses_result(cs, rs)
    sleep(10)
    print("Comparing...")
    # result = await compare_conv_summaries_embeddings(rs, cs, embed_fn)

    emb_client = EmbeddingClient()

    def compare_conv_summ_embeddings(a: TreatmentProposal, b: TreatmentProposal):
        return {
            "diagnoses_confirmed": emb_client.lists_similarity(a.diagnoses_confirmed, b.diagnoses_confirmed),
            "diagnoses_suspected": emb_client.lists_similarity(a.diagnoses_suspected, b.diagnoses_suspected),
            "treatment_plan": emb_client.lists_similarity(a.treatment_plan, b.treatment_plan),
            "risks": emb_client.lists_similarity(a.risks, b.risks),
            "next_steps": emb_client.lists_similarity(a.next_steps, b.next_steps),
            # "source_specialists": await list_embedding_similarity(a.source_specialists, b.source_specialists, embed_fn),
            "notes": emb_client.lists_similarity(a.notes, b.notes),
        }

    result = compare_conv_summ_embeddings(rs, cs)

    # Embeddings:: {'object': 'list', 'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.007914375, 0.004464019, -0.0035263803, -0.039486103, -0.0015487613, 0.0503401, -0.019259611, 0.045879982, 0.020054948, -0.0049006743 ...
    # overall = await overall_similarity_embeddings(summary1, summary2, token)

    print("===================")
    print("=======RESULT======")
    print(result)

    clean_cs = {k: json.dumps(v, ensure_ascii=False) for k, v in asdict(cs).items()}
    df1 = pd.DataFrame.from_dict(clean_cs, orient="index", columns=["conversation"])
    df1["conversation"] = df1["conversation"].apply(lambda x: json.loads(x) if isinstance(x, str) else str(x))
    df1["conversation"] = df1["conversation"].apply(lambda x: "\n ".join(x) if isinstance(x, list) else str(x))

    clean_rs = {k: json.dumps(v, ensure_ascii=False) for k, v in asdict(rs).items()}
    df2 = pd.DataFrame.from_dict(clean_rs, orient="index", columns=["reference"])
    df2["reference"] = df2["reference"].apply(lambda x: json.loads(x) if isinstance(x, str) else str(x))
    df2["reference"] = df2["reference"].apply(lambda x: "\n ".join(x) if isinstance(x, list) else str(x))

    df = df1.join(df2)
    print("DF::---------\n", df)

    df3 = pd.DataFrame.from_dict(result, orient="index", columns=["score"])
    df = df.join(df3)

    # zapisz do CSV
    df.to_csv(f"out-prot/csv-{conv_id}.csv", encoding="utf-8")


if __name__ == "__main__":
    # get_github_models()
    asyncio.run(main())
