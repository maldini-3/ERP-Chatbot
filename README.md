###RAG Chatbot with Qwen 7B Chat

Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot using Qwen 7B Chat for generating responses. It retrieves relevant ERP-related documents using FAISS and provides accurate answers based on the retrieved data.

Features

Retrieval-Based Search: Uses FAISS for efficient similarity search.

Text Generation: Utilizes Qwen 7B Chat for generating natural responses.

Preprocessed Data: Works with cleaned ERP metadata for better results.

Interactive Chat Interface: Command-line chat system for querying ERP knowledge.

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

Ensure you have the cleaned ERP metadata and FAISS index ready:

erp_metadata.json → Contains preprocessed ERP knowledge.

erp_data_faiss.index → FAISS index for fast retrieval.

4️⃣ Run the Chatbot

Execute the chatbot script:

python app.py

Then, start asking questions about ERP processes!

5️⃣ Example Queries

👤 You: كيف أجد فاتورة مبيعات؟
🤖 ERP Chatbot: يمكنك البحث عن فواتير المبيعات عبر قائمة "المحاسبة – الفواتير" أو باستخدام رقم الفاتورة.

File Structure

📂 RAG_Chatbot_Qwen7B
 ├── app.py               # Main chatbot script
 ├── Retrieval.py         # FAISS retrieval logic
 ├── embedding.py         # Embedding model for FAISS indexing
 ├── erp_metadata.json    # ERP-related metadata
 ├── erp_data_faiss.index # FAISS index file
 ├── README.md            # Project documentation

Future Improvements

Deploy chatbot using Streamlit for a web-based interface.

Fine-tune Qwen 7B Chat for ERP-specific terminology.

Optimize retrieval performance with advanced embedding models.

License

This project is open-source and free to use for research and development purposes.
