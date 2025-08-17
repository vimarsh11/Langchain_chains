from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

# Load env variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # lightweight + fast
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Plain text parser
parser = StrOutputParser()

# Define feedback schema
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Sentiment of the feedback"
    )

# Structured output parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Step 1: Sentiment classification
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text "
        "into positive or negative:\n\n{feedback}\n\n{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | llm | parser2

# Step 2a: Positive response
prompt2 = PromptTemplate(
    template="Write a warm, professional reply to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Step 2b: Negative response
prompt3 = PromptTemplate(
    template="Write a polite and empathetic reply to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# Conditional branching
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | llm | parser),
    (lambda x: x.sentiment == "negative", prompt3 | llm | parser),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)

# Final chain: classifier → branch
chain = classifier_chain | branch_chain

# Run it
if __name__ == "__main__":
    feedback = "The customer support team was super helpful and friendly!"
    result = chain.invoke({"feedback": feedback})
    print("\n=== Conditional Chain Output ===\n", result)

    # (Optional) visualize graph
    chain.get_graph().print_ascii()
