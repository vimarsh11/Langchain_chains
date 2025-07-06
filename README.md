# Langchain_chains
Here we are going to understand about chains in Langchain
<br>
| Tool                     | Purpose                                 |
| ------------------------ | --------------------------------------- |
| `langchain`              | Framework to build LLM apps             |
| `langchain-google-genai` | Gemini LLM wrapper                      |
| `python-dotenv`          | Loads `.env` files for API keys         |
| `StrOutputParser`        | Extracts clean output text from the LLM |

<b>simple_chain.py</b>
<br>
PromptTemplate formats the user input ({topic}) ->
ChatGoogleGenerativeAI sends it to Gemini Pro ->
StrOutputParser extracts and returns readable text ->
The result is printed!
