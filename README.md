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
<br>
<br>
<b>sequential_chain.py</b>
<br>
| Step             | Description                                         |
| ---------------- | --------------------------------------------------- |
| `RunnableMap`    | Runs `summary_chain` and stores output as `summary` |
| `RunnableLambda` | Prepares the output for next step                   |
| `explain_chain`  | Uses Gemini to simplify the summary                 |

![alt text](https://github.com/user-attachments/assets/fc86e9c9-f571-4f01-9acd-bfff1eaa79ff)
<br>

<b>parallel_chain.py<b>

![alt text](https://github.com/vimarsh11/Langchain_chains/blob/main/working%20_flow_parallel)

