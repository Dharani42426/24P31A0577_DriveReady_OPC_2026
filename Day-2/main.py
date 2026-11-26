
import os
import json
import functools

from openai import OpenAI, RateLimitError, APIConnectionError

# Connect to Groq
# Add GROQ_API_KEY to your environment before running.
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"

# Sample canteen menu
MENU = {
    "idli": 30,
    "veg sandwich": 40,
    "lemon juice": 25,
    "coffee": 20
}

# Convert menu dictionary into JSON text
menu_json = json.dumps(MENU, indent=2)

SYSTEM = f"""
ROLE: You are a friendly Campus Canteen Assistant.

FACTS:
Sample menu and prices in rupees:
{menu_json}

Sample opening hours: 8 AM to 5 PM.

RULES:
- Answer politely and clearly.
- Do not invent menu items or prices.
- If information is unknown, say
  "Please check with the canteen staff."
- Do not claim live stock availability.
- Keep answers under 50 words.
- Reply in English or Telugu matching the student.
"""

# Decorator to log function calls
def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"[log] {func.__name__} -> {result}")
        return result
    return wrapper


# Function to calculate item price
@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price for the selected menu item."""
    return MENU[item.lower().strip()] * qty


# Sample order using a Python dictionary
order = {
    "name": "Ravi",
    "item": "idli",
    "quantity": 2
}

# Convert order dictionary into JSON
order_json = json.dumps(order, indent=2)
print("Order JSON:")
print(order_json)

# Convert JSON back into a dictionary
order_data = json.loads(order_json)


# Chat history
history = [
    {"role": "system", "content": SYSTEM}
]

MAX_TURNS = 8


def chat(user_text: str) -> str:
    """Send a message to Groq and return the assistant reply."""

    history.append({
        "role": "user",
        "content": user_text
    })

    recent = [history[0]] + history[1:][-MAX_TURNS:]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=recent,
            temperature=0.3
        )

        reply = response.choices[0].message.content

    except RateLimitError:
        reply = "The service is busy. Please try again later."

    except APIConnectionError:
        reply = "Cannot connect to the server. Please try again."

    history.append({
        "role": "assistant",
        "content": reply
    })

    return reply


# Run the assistant
if __name__ == "__main__":

    print("Campus Canteen Assistant")

    print("Price of 2 Idli:", get_price("idli", 2))

    while True:
        user_text = input("You: ")

        if user_text.lower().strip() in ("bye", "exit", "quit"):
            print("Bot: Thank you for visiting!")
            break

        print("Bot:", chat(user_text))
