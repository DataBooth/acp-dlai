import marimo

__generated_with = "0.14.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Lesson 3 - Building a RAG Agent with CrewAI""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this lesson, you will build a RAG agent with `CrewAI`. You will do that by integrating `RagTool` from `crewai_tools` with a `CrewAI` agent. `RagTool` provides a way to create and query knowledge bases from various data sources, and allows the agent to access specialized context. 

    In this lesson, you will provide the RAG tool a pdf file containing details about insurance coverage provided by a private health insurer. By the end of the lesson, you will build an insurer agent specialized in answering queries related to health benefits. In the next lessons, you will wrap this agent in an ACP server and make it interact with other ACP agents.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>To Access <code>requirements.txt</code> and the <code>data</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.1. Import Libraries""")
    return


@app.cell
def _():
    from dotenv import load_dotenv
    return (load_dotenv,)


@app.cell
def _():
    from crewai import Crew, Task, Agent, LLM
    from crewai_tools import RagTool
    return Agent, Crew, LLM, RagTool, Task


@app.cell
def _():
    import warnings
    warnings.filterwarnings('ignore')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.2. Define the Agent's Large Language Model""")
    return


@app.cell
def _(load_dotenv):
    load_dotenv()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You'll now define the large language model that you will use for your CrewAI agent. `max_tokens`: maximum number of tokens the model can generate in a single response.

    **Note**: If you will define this model locally, it requires that you define the API key in a **.env** file as follows:
    ```
    # Required
    OPENAI_API_KEY=sk-...

    # Optional
    OPENAI_API_BASE=<custom-base-url>
    OPENAI_ORGANIZATION=<your-org-id>
    ```
    """
    )
    return


@app.cell
def _(LLM):
    llm = LLM(model="openai/gpt-4", max_tokens=1024)
    return (llm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.3. Define the RAG Tool""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For the RAG tool, you can define the model provider and the embedding model in a configuration Python dictionary. You can also define the details of your vector database. If you don't specify the vector database, the RagTool will use Chroma (ChromaDB) as the default vector database in local/in-memory mode.""")
    return


@app.cell
def _():
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4",
            }
        },
        "embedding_model": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002"
            }
        }
    }
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can then pass the `config` to the `RagTool`, and then specify the data source for which the knowledge base will be constructed. When embedding your data, the `RagTool` chunks your document into chunks and create an embedding vector for each chunk. You can specify the chunk size (`chunk_size`: number of characters) and how many characters overlap between consecutive chunks (`chunk_overlap`). You can also use the default behavior.""")
    return


@app.cell
def _():
    from pathlib import Path
    return (Path,)


@app.cell
def _(Path):
    (Path.cwd() / "data").exists()
    return


@app.cell
def _(Path, RagTool, config):
    rag_tool = RagTool(config=config,  
                       chunk_size=1200,       
                       chunk_overlap=200,     
                      )
    pdf_file = Path.cwd() / "data/gold-hospital-and-premium-extras.pdf"
    return pdf_file, rag_tool


@app.cell
def _(pdf_file):
    pdf_file
    return


@app.cell
def _(pdf_file, rag_tool):
    rag_tool.add(str(pdf_file), data_type="pdf_file")
    return


@app.cell
def _(rag_tool):
    rag_tool
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.4. Define the Insurance Agent""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that you have the `rag_tool` defined, you define the CrewAI agent that can assist with insurance coverage queries.""")
    return


@app.cell
def _(Agent, llm, rag_tool):
    insurance_agent = Agent(
        role="Senior Insurance Coverage Assistant", 
        goal="Determine whether something is covered or not",
        backstory="You are an expert insurance agent designed to assist with coverage queries",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool], 
        max_retry_limit=5
    )
    return (insurance_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.5. Define the Agent Task""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's now test the insurance agent. For that, you need to define the agent task and pass to it the query and the agent.""")
    return


@app.cell
def _(Task, insurance_agent):
    task1 = Task(
            description='What is the waiting period for rehabilitation?',
            expected_output = "A comprehensive response as to the users question",
            agent=insurance_agent
    )
    return (task1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.6. Run the Insurance Agent""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To run the agent, you need to pass the agent and the task to a Crew object that you can run using the `kickoff` method.""")
    return


@app.cell
def _(Crew, insurance_agent, task1):
    crew = Crew(agents=[insurance_agent], tasks=[task1], verbose=True)
    task_output = crew.kickoff()
    print(task_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3.7. Resources""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - [CrewAI Agents](https://docs.crewai.com/concepts/agents)
    - [CrewAI Tasks](https://docs.crewai.com/concepts/tasks)
    - [CrewAI RagTool](https://docs.crewai.com/tools/ai-ml/ragtool)
    - [Short course on Multi Agents with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
    <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>

    </div>
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
