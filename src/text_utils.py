
import json
from datasets import Dataset, load_dataset
import re
from pprint import pprint

DEFAULT_SYSTEM_PROMPT = """
Below is a conversation between a human and an AI agent. Write a summary of the conversation.
""".strip()

def generate_prompt(
        conversation: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> str:
        return f"""### Instruction: {system_prompt}
    
    ### Input:
    {conversation.strip()}
    
    ### Response:
    """.strip()

def clean_and_prepare(dataset):
    examples = []
    for data_point in dataset["test"].select(range(5)):
        summaries = json.loads(data_point["original dialog info"])["summaries"][
            "abstractive_summaries"
        ]
        summary = summaries[0]
        summary = " ".join(summary)
        conversation = create_conversation_text(data_point)
        examples.append(
            {
                "summary": summary,
                "conversation": conversation,
                "prompt": generate_prompt(conversation),
            }
        )
    test_df = pd.DataFrame(examples)
    return test_df

def summarize(model, text: str):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)


def generate_text(data_point):
    summaries = json.loads(data_point["original dialog info"])["summaries"][
        "abstractive_summaries"
    ]
    summary = summaries[0]
    summary = " ".join(summary)
 
    conversation_text = create_conversation_text(data_point)

    try:
        example = {
            "conversation": conversation_text,
            "summary": summary,
            "text": generate_training_prompt(conversation_text, summary),
        }
        return example

    except Exception as e:
        return {
            "conversation": "",
            "summary": "",
            "text": ""
        }

DEFAULT_SYSTEM_PROMPT = """
Below is a conversation between a human and an AI agent. Write a summary of the conversation.
""".strip()
 
 
def generate_training_prompt(
    conversation: str, summary: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    training_prompt = f"""
    ### Instruction: {system_prompt}
    ### Input:
    {conversation.strip()}

    ### Response:
    {summary}
    """.strip()

    return training_prompt

def create_conversation_text(data_point):
    text = ""
    for item in data_point["log"]:
        user = clean_text(item["user utterance"])
        text += f"user: {user.strip()}\n"
 
        agent = clean_text(item["system response"])
        text += f"agent: {agent.strip()}\n"
 
    return text
 
 
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_text)
        .remove_columns(
            [
                "original dialog id",
                "new dialog id",
                "dialog index",
                "original dialog info",
                "log",
                "prompt",
            ]
        )
    )