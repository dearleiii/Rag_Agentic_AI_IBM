"""
The prompt template and chain work together to create an efficient and automated workflow:

Prompt template: Provides a structured format for the model's input and ensures that the output aligns with the desired requirements.
Chain: Automates the process by linking raw input (e.g., the transcript) to the model through the template and handling the output.
Each step in the chain performs a specific function, such as:

Passing input data
Applying the prompt template
Invoking the language model and
Parsing the output into a usable form.
"""

# Define the prompt template
template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""
prompt = ChatPromptTemplate.from_template(template)

# Define the chain
chain = (
    {"context": RunnablePassthrough()}  # Pass the transcript as context
    | prompt
    | llm
    | StrOutputParser()
)