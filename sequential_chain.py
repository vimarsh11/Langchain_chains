from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompts
summary_prompt = PromptTemplate.from_template(
    "Give a 2-line summary of {topic}."
)

explain_prompt = PromptTemplate.from_template(
    "Explain this summary to a 5-year-old: {summary}"
)

# Output parser
parser = StrOutputParser()

# Chain 1: topic -> summary
summary_chain = summary_prompt | llm | parser

# Chain 2: summary -> explanation
explain_chain = explain_prompt | llm | parser

# Full chain: topic -> summary -> explanation
full_chain = (
    RunnableMap({
        "summary": summary_chain
    })
    | RunnableLambda(lambda d: {"summary": d["summary"]})
    | explain_chain
)

# Run it
if __name__ == "__main__":
    result = full_chain.invoke({"topic": "quantum physics"})
    print("\nFinal Output:\n", result)
