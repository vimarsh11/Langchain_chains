from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Define prompts
summary_prompt = PromptTemplate.from_template(
    "Give a 2-line summary of {topic}."
)

fun_fact_prompt = PromptTemplate.from_template(
    "Tell a fun fact about {topic}."
)

controversy_prompt = PromptTemplate.from_template(
    "What's a controversial opinion related to {topic}?"
)

# Output parser
parser = StrOutputParser()

# Individual chains
summary_chain = summary_prompt | llm | parser
fun_fact_chain = fun_fact_prompt | llm | parser
controversy_chain = controversy_prompt | llm | parser

# Parallel execution using RunnableMap
parallel_chain = RunnableMap({
    "summary": summary_chain,
    "fun_fact": fun_fact_chain,
    "controversy": controversy_chain
})

# Run it
if __name__ == "__main__":
    result = parallel_chain.invoke({"topic": "Artificial Intelligence"})
    print("\n=== Parallel Chain Output ===")
    for key, value in result.items():
        print(f"\n{key.capitalize()}:\n{value}")
parallel_chain.get_graph().print_ascii()