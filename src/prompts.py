from langchain_core.prompts import PromptTemplate

class Prompts:
    OUTLINES_GEN = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an outline generator agent. 
        You will be given a problem and your role is to generate a structured outline that will help you generate a professional report on the problem. 
        - The generated report should be professional, well structured, logical, and complete.
        - The outline only includes level 1 headings, do not generate level 2, level 3 headings.
        - The number of headings is less than 8.
        
        Please response in the following format:
        1. xxx
        2. xxx
        3. ...


        The Question: {question}
        Outline:"""
    )
    VALIDATE_RETRIEVAL = PromptTemplate(
        input_variables=["retrieved_context", "question"],
        template="""
        You are a retrieval validator.
        You will be provided with a question and chunks of text that may or may not contain the answer to the question.
        Your role is to carefullylook through the chunks of text provide a JSON response with three fields:
        1. status: whether the retrieved chunks contain the answer to the question.
        - 'COMPLETE' if the retrieved chunks contain the answer to the question, 'INCOMPLETE' otherwise. Nothing else.
        
        2. useful_information: the useful information from the retrieved chunks. Be concise and direct.
        - if there is no useful information, set this to an empty string.
        
        3. missing_information: the missing information that is needed to answer the question in full. Be concise and direct.
        - if there is no missing information, set this to an empty string.
        - if missing_information is not empty, 'status' must be 'INCOMPLETE'.
        
        Please provide your response as dictionary in the followingformat.

        {{"status": "<status>",
        "useful_information": "<useful_information>",
        "missing_information": "<missing_information>"}}
        
        Here is an example of the response format:
        
        {{"status": "COMPLETE",
        "useful_information": "The capital city of Canada is Ottawa.",
        "missing_information": "The capital city of Mexico"}}
    
        Do not include any other text.
        
        Context: {retrieved_context}
        
        The Question: {question}
        Response:
        """
    )
    ANSWER_QUESTION = PromptTemplate(
        input_variables=["retrieved_context", "question"],
        template="""
        You are a deep research writing agent.
        You will be provided with a topic and chunks of text that contain the related information to the topic.
        Your role is to carefully look through the chunks of text and write a deep research about the topic.
        
        Here is the requirement of your writing:
        1. Don't include the topic title or try to write other sections, only write the article content without any formatting.
        2. Please generate 3-5 paragraphs. Each paragraph is at least 500 words long.
        3. Please ensure that the data of the article is true and reliable, the logical structure is clear, the content is complete, and the style is professional, so as to attract readers to read.
        
        The topic: {question}
        Context: {retrieved_context}
        writing:
        """
    )
