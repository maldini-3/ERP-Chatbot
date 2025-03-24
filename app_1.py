from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from Retrieval import retrieve_relevant_chunks

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def generate_response(user_query):
    retrieved_chunks = retrieve_relevant_chunks(user_query)
    best_chunk = retrieved_chunks[0] if retrieved_chunks else ""
    
    if not best_chunk.strip():
        return "❌ لم أجد معلومات ذات صلة بسؤالك."
    
    prompt = f"""
    المستخدم: {user_query}
    
    المعلومات المتاحة:
    {best_chunk.strip()}
    
    المساعد: بناءً على المعلومات أعلاه، أجب بإيجاز ودقة دون إضافة معلومات غير مذكورة.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    output = model.generate(
        **inputs, max_new_tokens=150, repetition_penalty=1.2, temperature=0.7, top_p=0.9, do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    print("🟢 يمكنك الآن كتابة أسئلتك (اكتب 'exit' للخروج)")
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 وداعًا!")
            break
        elif user_input:
            response = generate_response(user_input)
            print(f"\n----------------------\n🤖 ERP Chatbot: {response}\n----------------------\n")
        else:
            print("⚠️ الرجاء إدخال سؤال صالح.")
