def remove_non_ascii(text):
    """
    This function removes all characters outside the ASCII range from a given text. ASCII characters include standard English letters, numbers, and symbols. It ensures compatibility with systems or models that only process ASCII characters and helps avoid errors caused by unsupported symbols such as emojis or accented letters.
    """
    return ''.join(i for i in text if ord(i) < 128)


def product_assistant(ascii_transcript):
    """
    The product_assistant function processes transcripts of the sample earnings call to ensure that financial product 
    terms are correctly formatted. 
    It uses a detailed system prompt to transform terms (e.g., '401k' to '401(k) retirement savings plan') and 
    resolves contextual ambiguities for acronyms such as 'LTV'. 
    
    Note that in this project, this assistant is specifically designed to work with financial terminology. 
    However, you can customize its prompts to suit your unique use case.
    """
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)' , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)', and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing financial products into their numeric representations, followed by the full name of the product in parentheses. For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'. However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to  and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted transcript and a list of the words you've changed"""

    # Concatenate the system prompt and the user transcript
    prompt_input = system_prompt + "\n" + ascii_transcript

    # Create a messages object
    messages = [
        {
            "role": "user",
            "content": prompt_input
        }
    ]

    # Construct the model ID using the specified model size
    model_id = f"meta-llama/llama-3-2-11b-vision-instruct"
    
    # Configure the parameters for model behavior
    params = TextChatParameters(
        temperature=0.2,  # Controls randomness; lower values make the output more deterministic
        top_p=0.6              # Nucleus sampling to control output diversity
    )
    
    # Initialize the Llama 3.2 model inference object
    llama32 = ModelInference(
        model_id=model_id,         # Specify the model ID to use (Llama 3.2)
        credentials=credentials,   # Authentication credentials for accessing the model
        project_id=project_id,     # Link to the associated project ID
        params=params              # Parameters that define the model's response behavior
    )
    
    # Send the input messages to the model and retrieve its response
    response = llama32.chat(messages=messages)
    
    # Extract and return the content of the model's first response choice
    return response['choices'][0]['message']['content']