evaluate_system = """
You are an expert evaluator with extensive experience in evaluating response of given query.
""".strip()

criteria_gen_prompt = """Please generate five strict evaluation criteria for assessing the response given the following query. Each criterion should include the following fields: name, criteria_description, 1-2, 3-4, 5-6, 7-8, 9-10.
The criteria should be designed to emphasize detailed assessment and distinguish subtle differences in quality. Ensure that the criteria can discern issues such as relevance, coherence, depth, specificity, and adherence to the query context.
Do not include any additional text. Only output the criteria in the specified JSON format.

** Query **
{query}

** Output format **
```json
[
    {{
        "name": "first_criteria_name",
        "criteria_description": "Description for the first criteria, emphasizing detailed and critical assessment.",
        "1-2": "Low score description: Clearly deficient in this aspect, with significant issues.",
        "3-4": "Below average score description: Lacking in several important areas, with noticeable problems.",
        "5-6": "Average score description: Adequate but not exemplary, meets basic expectations with some minor issues.",
        "7-8": "Above average score description: Generally strong but with minor shortcomings.",
        "9-10": "High score description: Outstanding in this aspect, with no noticeable issues." 
    }},
    ...
]
```
"""

evaluate_prompt = """
Evaluate the Response based on the Query and criteria provided. Please response in chinese.

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria:

```{criteria}```

Provide reasons for each score, indicating where and why any strengths or deficiencies occur within the Response. Reference specific passages or elements from the text to support your justification.
Ensure that each reason is concrete, with explicit references to the text that aligns with the criteria requirements.

Scoring Range: Assign an integer score between 1 to 10

** Output format **
Return the results in the following JSON format, Only output this JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```
""".strip()
