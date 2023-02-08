import openai
from util import get_user_input
openai.api_key = "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"

prompt = get_user_input('enter prompt', [])
print("You Entered:", prompt)

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.6,
)
print("response:", response.choices[0].text)
