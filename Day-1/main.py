
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from google.colab import userdata

# Get Groq API key from Google Colab Secrets
GROQ_API_KEY = userdata.get("GROQ_API_KEY")

# Create the language model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)

# Instructions for our canteen assistant
SYSTEM_PROMPT = """
You are a helpful Campus Canteen Assistant.

Your job is to answer students' questions about
canteen food items, prices and timings.

Sample menu:
- Idli: Rs. 30
- Veg Sandwich: Rs. 40
- Lemon Juice: Rs. 25
- Coffee: Rs. 20

Rules:
1. Answer politely and clearly.
2. Do not invent menu items or prices.
3. These are sample menu details.
4. If you do not know an answer, say
   "Please check with the canteen staff."
5. Keep answers short and simple.
"""

# Function to ask questions
def ask(question):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    return response.content


# Test the assistant
if __name__ == "__main__":
    question = "What is the price of Idli?"
    answer = ask(question)

    print("Student:", question)
    print("Assistant:", answer)
