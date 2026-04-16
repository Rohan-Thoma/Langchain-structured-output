from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict

#define our normal model as usual
model = ChatOpenAI()

#lets take this review 
review = """The hardware is great, but the software feels bloated. There are too many pre-installed\
      apps that I can't remove. Also, the UI looks outdated compared to other brands.\
          For a software update fix this
"""

#now, before passing this review to the model, we will first define a typed-dict
class Review(TypedDict):
    summary: str
    sentiment: str

#now we will convert this above defined model into a structured output model
structured_model = model.with_structured_output(Review)

#Now we will invoke this structured model now
result = structured_model.invoke(review)


#lets print the result and see for ourselves
print(result)

print("\n=============================\n")

print(type(result))