RAG Chatbot with Qwen 7B Chat

Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot powered by Qwen 7B Chat for generating accurate responses. It combines information retrieval with natural language generation to provide intelligent and context-aware answers.

Features

Retrieval-Based Search: Uses FAISS for efficient similarity search.

Text Generation: Utilizes Qwen 7B Chat for generating natural responses.

Preprocessed Data: Works with cleaned metadata for better results.

Interactive Chat Interface: Command-line chat system for querying various knowledge sources.

Setup Instructions

1️⃣ Install Dependencies

Make sure you have the necessary Python libraries installed:

pip install torch transformers sentence-transformers faiss-cpu streamlit

2️⃣ Download the Model

The chatbot uses Qwen 7B Chat, which can be downloaded from Hugging Face:

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

3️⃣ Prepare the Data

Ensure you have the cleaned metadata and FAISS index ready:

metadata.json → Contains preprocessed knowledge.

data_faiss.index → FAISS index for fast retrieval.

4️⃣ Run the Chatbot

Execute the chatbot script:

python app.py

Then, start asking questions on various topics!

5️⃣ Example Queries

👤 You: ما هي القوائم المالية؟
🤖 Chatbot: القوائم المالية تشمل قائمة الدخل والميزانية العمومية والتدفقات النقدية.

File Structure

📂 RAG_Chatbot_Qwen7B
 ├── app.py               # Main chatbot script
 ├── Retrieval.py         # FAISS retrieval logic
 ├── embedding.py         # Embedding model for FAISS indexing
 ├── metadata.json        # Knowledge metadata
 ├── data_faiss.index     # FAISS index file
 ├── README.md            # Project documentation

Future Improvements

Deploy chatbot using Streamlit for a web-based interface.

Fine-tune Qwen 7B Chat for specific domains.

Optimize retrieval performance with advanced embedding models.

License

This project is open-source and free to use for research and development purposes.
