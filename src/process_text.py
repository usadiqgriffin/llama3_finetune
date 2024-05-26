import logging
import json
import re
from pprint import pprint
#from src import process_text, model

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import logging

DEFAULT_SYSTEM_PROMPT = """
                        Below is a conversation between a human and an AI agent. Write a summary of the conversation.
                        """.strip()

def generate_training_prompt(
    conversation: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    training_prompt = f"""### Instruction: {system_prompt}

                    ### Input:
                    {conversation.strip()}

                    ### Response:
                    {summary}
                    """.strip()
    
    return training_prompt

def generate_deployment_prompt(
    conversation: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    deployment_prompt = f"""### Instruction: {system_prompt}

    ### Input:
    {conversation.strip()}

    ### Response:
    """.strip()

    return deployment_prompt

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\^[^ ]+", "", text)
    return text


def create_conversation_text(data_point):
    text = ""
    for item in data_point["log"]:
        user = clean_text(item["user utterance"])
        text += f"user: {user.strip()}\n"

        agent = clean_text(item["system response"])
        text += f"agent: {agent.strip()}\n"

    return text

def generate_text(data_point):
    summaries = json.loads(data_point["original dialog info"])["summaries"][
        "abstractive_summaries"
    ]
    summary = summaries[0]
    summary = " ".join(summary)

    conversation_text = create_conversation_text(data_point)
    text = {
        "conversation": conversation_text,
        "summary": summary,
        "text": generate_training_prompt(conversation_text, summary),
    }

    return text

def process_dataset(data: Dataset):

    logging.info("Processing dataset...")
    data =  data.shuffle(seed=42).map(generate_text)
    data = data.remove_columns(
            [
                "original dialog id",
                "new dialog id",
                "dialog index",
                "original dialog info",
                "log",
                "prompt",
            ])
    return data

