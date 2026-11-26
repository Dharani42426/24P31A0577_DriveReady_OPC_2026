
# Install first:
# pip install langchain langchain-groq

import os
import csv

from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser

# Set GROQ_API_KEY in your environment
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Please set your GROQ_API_KEY first.")

# Connect to the language model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)

# Campus Canteen Assistant instructions
SYSTEM = """
ROLE: You are a friendly Campus Canteen Assistant.

FACTS:
Sample menu:
Idli Rs. 30
Veg Sandwich Rs. 40
Lemon Juice Rs. 25
Coffee Rs. 20

Sample opening hours: 8 AM to 5 PM.

RULES:
- Answer politely and clearly.
- Do not invent menu items or prices.
- Do not claim live stock availability.
- If information is unknown, say
  "Please check with the canteen staff."
- Work out totals carefully.

FORMAT:
Keep answers simple and under 50 words.
Reply in English or Telugu matching the student.
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

# Create LangChain chain
chain = prompt | llm | StrOutputParser()

# Store conversation history
history = []


def ask(question: str) -> str:
    """Ask a question and remember the conversation."""

    answer = chain.invoke({
        "history": history,
        "question": question
    })

    history.extend([
        ("human", question),
        ("ai", answer)
    ])

    return answer


# Test the assistant using CSV questions
def run_tests(path: str = "tests.csv") -> None:
    """Run CSV tests and calculate the pass percentage."""

    passed = 0

    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        answer = chain.invoke({
            "history": [],
            "question": row["question"]
        })

        expected = row["must_include"].lower()
        ok = expected in answer.lower()

        if ok:
            passed += 1

        print(
            "PASS" if ok else "FAIL",
            "|",
            row["question"],
            "->",
            answer[:70]
        )

    total = len(rows)

    if total:
        print(
            f"\n{passed}/{total} passed "
            f"({passed / total:.0%})"
        )
    else:
        print("No tests found.")


# Run the assistant
if __name__ == "__main__":
    print("Campus Canteen Assistant")

    print(ask("What is the price of Idli?"))
    print(ask("What is my previous question?"))

    print("\nRunning tests...")
    run_tests()
