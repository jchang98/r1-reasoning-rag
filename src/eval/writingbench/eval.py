import json
import sys
from tqdm import tqdm
import argparse
from src.eval.writingbench.prompt import evaluate_system, evaluate_prompt, criteria_gen_prompt
from src.eval.writingbench.critic import CriticAgent
from src.eval.writingbench.llm import ClaudeAgent
import os

EVAL_TIMES = 1


class EvalAgent(object):
    def __init__(self, agent):
        self.agent = agent
    
    def success_check_fn_score(self, response):
        try:
            result = json.loads(response.strip('json|```'))
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            return False
        
        valid_score_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        if "score" not in result or "reason" not in result:
            print("Missing 'score' or 'reason' in the result")
            return False
        if result["score"] not in valid_score_values:
            return False
        if not isinstance(result["reason"], str):
            return False
        return True
    
    def generate_score(self, response, query, criteria):
        prompt_data = {
            "query": query,
            "response": response,
            "criteria": criteria,
        }
        retry = 0
        success = False
        while not success and retry < 3:
            prompt = evaluate_prompt.format(**prompt_data)
            response, success = self.agent.run(
                prompt=prompt,
                success_check_fn=self.success_check_fn_score
            )
            try:
                response = json.loads(response.strip('json|```'))
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                response = eval(response.strip('json|```'))
            retry += 1
        if success:
            return response
        else:
            raise ValueError("Fail to generate score!")
    
    def generate_criteria(self, query):
        prompt_data = {
            "query": query,
        }
        retry = 0
        success = False
        while not success and retry < 3:
            prompt = criteria_gen_prompt.format(**prompt_data)
            response, success = self.agent.run(
                prompt=prompt
            )
            try:
                response = json.loads(response.strip('json|```'))
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                response = eval(response.strip('json|```'))
            retry += 1
        if success:
            return response
        else:
            raise ValueError("Fail to generate score!")


def save_output(output, file_name):
    """
    Saves output data to a specified file in JSONL format.
    """
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    """
    Loads JSONL lines from a file into a list of dictionaries.
    """
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

def gen_criteria(agent, input_file, query_criteria_file):
    """
        Generates criteria based on input queries.
    """

    contents, input_cnt = load_file(input_file)
    criteria_map = []
    cnt = 0
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {input_file.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            query = content['query']
            response = content['response']

            criteria = agent.generate_criteria(query)

            criteria_map.append(
                {
                    "query": query,
                    "criteria": criteria
                }
            )

            cnt += 1
            pbar.update()

        print(f"CNT: {cnt}")
    with open(query_criteria_file, 'w') as f:
        json.dump(criteria_map, f, ensure_ascii=False, indent=4)
    return 
            



def process(agent, input_file, out_file, id_query_criteria_map):
    """
    Processes input files through the evaluation agent, producing scores and saving results.
    """

    print("==== step2 ====\nproducing scores.")
    contents, input_cnt = load_file(input_file)
    cnt = 0
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {input_file.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            data = {
                "index": i,
                "scores": {}
            }

            query = content['query']
            response = content['response']
            criteria = [t_criteria for t_criteria in id_query_criteria_map if t_criteria["query"] == query][0]["criteria"]

            with tqdm(total=len(criteria) * EVAL_TIMES, desc=f"Data ID {i} Progress", leave=False) as internal_pbar:
                for c in criteria:
                    if c["name"] not in criteria:
                        data["scores"][c["name"]] = []
                    while len(data["scores"][c["name"]]) < EVAL_TIMES:
                        score = agent.generate_score(response, query, c)
                        data["scores"][c["name"]].append(score)
                        internal_pbar.update(1)

            save_output([data], out_file)
            cnt += 1
            pbar.update()

        print(f"CNT: {cnt}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--evaluator", choices=['claude', 'critic'], default="claude", help="Choose the scoring model to use: 'claude' or 'critic'.")
    parser.add_argument("--query_criteria_file", type=str, default="/mnt/data/jinchang/smolagents/r1-reasoning-rag-v5_1/r1-reasoning-rag/src/eval/writingbench/r1_criteria.jsonl", help="Path to the query and criteria file.")
    parser.add_argument("--input_file", default="/mnt/data/jinchang/smolagents/r1-reasoning-rag-v5_1/r1-reasoning-rag/src/eval/writingbench/query_pare_all.jsonl", type=str, help="Path to the input file.")
    parser.add_argument("--output_file", default="/mnt/data/jinchang/smolagents/r1-reasoning-rag-v5_1/r1-reasoning-rag/src/eval/writingbench/score_r1.jsonl", type=str, help="Path to the output file.")

    args = parser.parse_args()

    # Evaluator initialization based on chosen model
    if args.evaluator == 'claude':
        agent = EvalAgent(ClaudeAgent(
            system_prompt=evaluate_system,
        ))
    else:
        agent = EvalAgent(CriticAgent(
            system_prompt=evaluate_system,
        ))

    # Load query 
    # 根据query生成criteria
    if not os.path.isfile(args.query_criteria_file):
        print("==== step1 ====\nGenerates criteria based on input queries.")
        id_query_criteria_map = gen_criteria(agent, args.input_file, args.query_criteria_file)
    else:
        print("==== step1 ====\nLoad Already criteria files.")

    with open(args.query_criteria_file, 'r') as f:
        id_query_criteria_map = json.load(f)


    process(agent, args.input_file, args.output_file, id_query_criteria_map)
