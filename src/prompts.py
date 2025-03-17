from langchain_core.prompts import PromptTemplate
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d")

class Prompts:
    CLARIFY = PromptTemplate(
        input_variables=["question", "now"],
        template="""
        You are a clarify agent. Today is {now}.

        Analyze the research topic: "{question}" and identify any ambiguous or unclear aspects that need clarification. Generate up to 5 clarifying questions that will help better understand the user's research intent.
        Requirements for the follow-up questions:
        - Focus on ambiguous or undefined aspects in the original query.
        - Please prompt the user to help narrow down the user's intent.
        - If the original query is sufficiently clear and comprehensive, you may return an empty string.
        - Return the response as raw text string.

        The topic: {question}
        Clarifying Questions:
        """
    )
    GEN_FC_QUERY = PromptTemplate(
        input_variables=["question", "now"],
        template="""
        You are an expert assistant who can solve any task using  tool calls. You will be given a task to solve as best you can. Today is {now}.
        To do so, you have been given access to the following tools: 
        {{"name": "search", "description": "useful for when you need to answer questions about current events, news."}}
        {{"name": "data_agent", "description": "useful for when you need to get financial data, including company and industry financial data."}}


        The tool call you write is an action list, Here are a few examples using notional tools:

        Task: "Which city has the highest population , Guangzhou or Shanghai?"

        Action List:
        ```json
        [
        {{'name': 'search', 'question': 'Population Guangzhou'}},
        {{'name': 'search',  question': 'Population Shanghai'}},
        ]
        ```

        ---
        Task: "Which company has a higher market capitalization, Tonghuashun or Dongfang Caifu?"

        Action List:
        ```json```
        [
        {{'name': 'data_agent', 'question': 'Tonghuashun market capitalization'}},
        {{'name': 'data_agent', 'question': 'Dongfang Caifu market capitalization'}},
        ]
        ```

        Here are the rules you should always follow to solve your task:
        1. ALWAYS provide a tool call, else you will fail.
        2. Call a tool only when needed: do not call the search if you do not need information, try to solve the task yourself.
        3. Never re-do a tool call that you previously did with the exact same parameters.
        4. Please response in the examples format, returns a json list
        5. The len of action list cannot exceed 5. 
        6. please response in chinese.

        Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.


        Task: {question}
        Action List:
        """
    )

    OUTLINES_GEN = PromptTemplate(
        input_variables=["question", "max_outline_num", "now"],
        template="""
        You are an outline generator agent. Today is {now}.
        You will be given a problem and your role is to generate a structured outline that will help you generate a professional report on the problem. 
        - Please clarify the user's needs according to the user's input questions, so that the generated outline meets the user's needs.
        - The generated report should be professional, well structured, logical, and complete.
        - The outline includes level 1 'headings' and the 'research_goal' corresponding to level 1 headings.
        - The number of headings cannot exceed {max_outline_num}.
        - Do not generate a 'headings' like 'conclusion' in the generated outline.
        - please response in chinese.
        
        Please response in the following format, returns a json list:
        ```json
        [
        {{'no': 1, 'headings': 'xxx', 'research_goal': 'xxx'}},
        {{'no': 2, 'headings': 'xxx', 'research_goal': 'xxx'}},
        ...
        ]
        ```


        The Question: {question}
        Outline:"""
    )

    GEN_USEFUL_INFO = PromptTemplate(
        input_variables=["retrieved_context", "question", "now"],
        template="""
    You are a useful information generator agent.  Today is {now}.  
    You will be provided with a question, chunks of text that may or may not contain the answer to the question, and the corresponding citations for each chunk of text.  Your role is to carefully review the chunks of text and extract the useful information from the retrieved chunks.  Additionally, you must provide the corresponding citations(webpage url) for the useful information.  
    
    Here are the requirements:
    - Remember, you must return both useful information and citations. Retain the original citations.
    - If there is no useful information, set this to an empty json list.
    - Please retain the information subject in the original information, as well as digital information, news policy information, reported news, etc.
    - Please retain the information that you don't know.
    - Return the useful information in an extremely detailed version, and return the additional context (if relevant).
    - Please make sure that the numerical values, news information in the 'Useful Information' are all from the 'Context'.
    - Please provide your response as a dictionary in the following format:


    Please response in the following format, returns a json list:
        ```json
        [
        {{'useful_information': 'xxx', 'url': 'xxx'}},
        {{'useful_information': 'xxx', 'url': 'xxx'}},
        ...
        ]
        ```

    Here is an example of the response format:
        
        ```json
        [
        {{'useful_information': '"The capital city of Mexico is Mexico City..."', 'url': 'https://simple.wikipedia.org/wiki/Mexico_City'}},
        ...
        ]
        ```

    Context: {retrieved_context}
    The Question: {question}
    Useful Information:"""
    )
    VALIDATE_RETRIEVAL = PromptTemplate(
        input_variables=["useful_information", "question", "now"],
        template="""
        You are a retrieval validator. Today is {now}.
        You will be provided with a question and chunks of text that may or may not contain the answer to the question.
        Your role is to carefullylook through the chunks of text provide a JSON response with two fields:
        1. status: whether the retrieved chunks contain the answer to the question.
        - 'COMPLETE' if the retrieved chunks contain the answer to the question, 'INCOMPLETE' otherwise. Nothing else.
                
        2. missing_information: the missing information that is needed to answer the question in full. Be concise and direct.
        - if there is no missing information, set this to an empty string.
        - if 'missing_information' is not empty, 'status' must be 'INCOMPLETE'; otherwise, if 'missing_information' is empty, 'status' must be 'COMPLETE'.
        
        Please provide your response as dictionary in the followingformat.

        {{"status": "<status>",
        "missing_information": "<missing_information>"}}
        
        Here is an example of the response format:
        
        {{"status": "COMPLETE",
        "missing_information": "The capital city of Mexico"}}
    
        Do not include any other text. please response in chinese.
        
        Context: {useful_information}
        
        The Question: {question}
        Response:
        """
    )
    ANSWER_QUESTION = PromptTemplate(
        input_variables=["useful_information", "question", "now"],
        template="""
        You are a deep research writing agent. Today is {now}.
        You will be provided with a topic and chunks of text that contain the related information to the topic.
        Your role is to carefully look through the chunks of text and write a deep research about the topic.
        
        Here is the requirement of your writing:
        1. Don't start your writing with '# title' or try to write other sections, only write the article paragraph content directly.
        2. Please generate 3-5 paragraphs. Each paragraph is at least 500 words long.
        3. Please ensure that the data of the article is true and reliable, the logical structure is clear, the content is complete, and the style is professional, so as to attract readers to read.
        4. If the 'Context' includes structured information, you can mix with charts in output writing. Please use the grammar of markdown to generate tables and the grammar of mermaid to generate pictures (including mind maps, flow charts, pie charts, gantt charts, timeline charts, etc.)
        5. Please make sure that the numerical value, news information in the output 'writing' are all from the 'Context'.
        6. For keywords information in output 'writing' paragraph content please use the markdown syntax ( **xxx**) to make it bold.
        7. Not all content in the 'Context' is closely related to the user's topic. You need to evaluate and filter the 'Context' based on the topic.
        8. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        9. Do not include any other text than the paragraph content. please response in chinese.
        
        Context: {useful_information}

        The topic: {question}
        Writing:
        """
    )
    ANSWER_QUESTION_SERIAL = PromptTemplate(
        input_variables=["useful_information", "question", "already_writing", "now"],
        template="""
        You are a deep research writing agent. Today is {now}.
        You will be provided with a topic, the already written text and chunks of text that contain the related information to the topic.
        Your role is to carefully look through the chunks of text and write a deep research about the topic.
        
        Here is the requirement of your writing:
        1. Don't start your writing with '# title' or try to write other sections, only write the article paragraph content directly.
        2. Please generate 3-5 paragraphs. Each paragraph is at least 500 words long.
        3. Please ensure that the data of the article is true and reliable, the logical structure is clear, the content is complete, and the style is professional, so as to attract readers to read.
        4. If the 'Context' includes structured information, you can mix with charts in output writing. Please use the grammar of markdown to generate tables and the grammar of mermaid to generate pictures (including mind maps, flow charts, pie charts, gantt charts, timeline charts, etc.)
        5. Please make sure that the numerical value, news information in the output 'writing' are all from the 'Context'.
        6. For keywords information in output 'writing' paragraph content please use the markdown syntax ( **xxx**) to make it bold.
        7. Maintain narrative consistency with previously written sections while avoiding content duplication. Ensure smooth transitions between sections.
        8. Not all content in the 'Context' is closely related to the user's topic. You need to evaluate and filter the 'Context' based on the topic.
        9. Use [1], [2], ..., [n] in line (for example, "The capital of the United States is Washington, D.C.[1][3]."). You DO NOT need to include a References or Sources section to list the sources at the end.
        10. Do not include any other text than the paragraph content. please response in chinese.
        
        Context: {useful_information}

        Already written text:
        {already_writing}

        The topic: {question}
        Writing:
        """
    )
    POLIESH_ANSWER = PromptTemplate(
        input_variables=["question", "outlines", "draft_writing", "now"],
        template="""
        You are a profession writing agent. Today is {now}.
        You won't delete any non-repeated part in the draft article. You will keep the charts(mermaid and markdown table code block) and draft article structure (indicated by "#") appropriately. Do your job for the following draft article.

    
        Here is the format of your writing:
        - Please remove repetitive part in the draft article and ensure that the transitions between different chapters are smoother for better understanding.
        - Be sure to save the non-repetitive parts of the draft article and only optimize for coherence.
        - Do not include any other text. please response in chinese.


        The outlines of the article:
        {outlines}

        The draft article:
        {draft_writing}

        The topic you want to write:{question}
        Polished article:
        """
    )
    CONCLUSION = PromptTemplate(
        input_variables=["question", "report", "now"],
        template = """
        # The following contents are the written article related to the user's question:
        {report}


        You now need to generate a conclusion based on the written article. When responding, please keep the following points in mind:
        - Today is {now}.
        - Use "#" Title" to indicate conclusion title, don't generate a "##" Title.
        - You need to interpret and summarize the user's requirements, choose an appropriate format, fully utilize the 'written article', extract key information, and generate an answer that is insightful, creative, and professional. Extend the length of your response as much as possible, addressing each point in detail and from multiple perspectives, ensuring the content is rich and thorough.
        - For analytical essay tasks, please provide insight and perspective on the topic, event, or phenomenon based on the user's question and written article.
        - For predictive essay tasks: It is necessary to predict and speculate on future development based on existing data and trends.
        - For explanatory essay tasks: You need to explain a concept, process, or phenomenon in detail, usually providing information in a clear, concise manner.
        - For argumentative essay tasks: Please present an argument and support it with evidence and logical reasoning designed to convince the reader.
        - For commentary essay tasks: You need to provide personal opinions and evaluations on a specific event, work, or phenomenon, often reflecting subjective viewpoints.
        - For comparative essay tasks: You need to compare and contrast two or more topics, events, or phenomena, highlighting their similarities and differences.
        - For review essay tasks: You need to offer a retrospective look at the historical development of a particular field or topic, summarizing key milestones and trends.
        - plase response in chinese.


        # The user's question is:
        {question}"""
    
    )
