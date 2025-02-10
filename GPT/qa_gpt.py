from transformers import pipeline, GPT2Tokenizer

# GPT-2 tabanlı bir metin üretme pipeline'ı oluşturuyoruz
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
qa_pipeline = pipeline("text-generation", model="gpt2")

def ask_gpt(question, max_length=100):
    """
    GPT modeline bir soru sorar ve yanıt döndürür.
    :param question: Kullanıcının sorduğu soru (string)
    :param max_length: Üretilen metnin maksimum uzunluğu (varsayılan: 100)
    :return: Modelin ürettiği yanıt
    """
    response = qa_pipeline(question, max_length=max_length, do_sample=True, temperature=0.7, truncation=True, pad_token_id=tokenizer.eos_token_id)
    return response[0]['generated_text']

# Test
if __name__ == "__main__":
    question = input("Sorunuzu girin: ")
    answer = ask_gpt(question)
    print("Yanıt:", answer)