import os

from groq import Groq
from constants import GROQ_API_KEY

client = Groq(
    api_key=GROQ_API_KEY,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "tell me about India",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)