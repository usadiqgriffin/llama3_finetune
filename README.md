## **Meta's latest LLM llama3 released and available to users through HuggingFace API**

### **Introduction**
In this project, I used a pre-trained quantized version of llama3-base on lavita/ChatDoctor-HealthCareMagic-100k data set and used PEFT/LoRA to reduce update size and help fit model into memory. 

[](images/image.png)

The model is then fine-tuned on a Lightning.ai T4 GPU (24GB) RAM where it trained without any issue.

<p align="center">
<img src="https://imageio.forbes.com/specials-images/imageserve/662202846cb84bb3a193445a/llama3/960x0.png?format=png&width=1440" width="400">
</p>

### **Installation, fine-tuning and deployment**

See [Llama3 Chat Doctor Notebook](https://github.com/usadiqgriffin/llama3_finetune/blob/main/llama3_chat_doctor_qa_fine_tune.ipynb)

### **Sample AI Doctor Response After Fine-Tune**

**Patient's Query**
I am a male 79 years old, have brief chills the may sweat have had for about 2 months also white
blood cell count to high had bone marrow test and ok so far dr has no idea what is wrong when I
chill have no fever as I check that and blood pressure. thanks

**AI response**
Thank you for sharing your symptoms with me. It's good to know that your blood work and bone marrow
test came back normal.
Based on your symptoms of brief chills and sweating, I would like to investigate further. I would
like to perform a thorough physical examination, including checking your vital signs, such as your
blood pressure, pulse, and respiratory rate.
Additionally, I would like to order some additional tests to help us better understand what's going
on. These tests may include a complete blood count, a urinalysis, and possibly some imaging
studies, such as a chest X-ray or a computed tomography (CT) scan.
