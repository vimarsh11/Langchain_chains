from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Define prompt template
prompt = PromptTemplate.from_template("What are 3 key facts about {topic}?")

# Define parser
parser = StrOutputParser()

# Build the chain
chain = prompt | llm | parser

# Run the chain
if __name__ == "__main__":
    result = chain.invoke({"topic": "black holes"})
    print(result)
chain.get_graph().print_ascii()