import json
import ollama
import torch
from Employee_data import employee_data, allocation_data
from Prompt import COT_prompt
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

class EmployeeFinder:
    def __init__(self) -> None:
        
        # self.client = Groq(api_key="YOUR_API_KEY")
        # self.model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
        self.employee_data = employee_data
        self.all_employee_categories = list(self.employee_data.keys())
        print(f"Total Departments: {self.all_employee_categories}")
        print("*"*20)
        
        self.COT_prompt = COT_prompt
        
        
    # def branch(self, prompt):
    #     branches = []
        
    #     for i in range(len(self.all_employee_categories)):
    #         response = self.reasoner(COT_prompt, prompt)

        
        
    def __call__(self, msg: str) -> str:
        
        category = self.get_relevant_category(msg)
        # Retrieve the list of employees for that category.
        candidates = self.employee_data.get(category)
        
        if candidates:
            
        
            # Create a nicely formatted string version of the candidate records.
            candidate_str = json.dumps(candidates, indent=2)

            # Build a prompt that includes the candidate data and current allocations.
            system_prompt = (
                "You are an AI HR assistant. Your task is to allocate workers to projects "
                "based on skills, experience, and current workload. "
                "If no perfect match exists, suggest candidates that can be quickly trained. "
                "\n\nCandidate Data:\n"
                f"{candidate_str}\n\n"
                "Current Allocations (do not reassign these workers):\n"
                f"{json.dumps(allocation_data, indent=2)}\n\n"
                "Output your response in the following structured format using one line per candidate:\n"
                "Optimal(<name>): <description>\n"
                "Suggestion(<name>): <description>\n"
            )

            
            response = self.llm_response(system_prompt, msg)
            return response
        
        else:
            # thus it means no department for that job position found
            # thus it means no employee has that experience
            prompt = f"""I want an employee who can be a good candidate '{msg}'\n
            The Employees in the Company are:{json.dumps(employee_data)}\n
            The Employee Allocation Data is here:{json.dumps(allocation_data)}\n
            lets think step by step,
            """
            response = self.reasoner(self.COT_prompt, prompt)
            return response


    def get_relevant_category(self, msg: str) -> str:
        # Use semantic similarity to pick the most relevant category key.
        # Encode the input message and all available categories.
        # embs_msg = self.model.encode([msg])
        # embs_categories = self.model.encode(self.all_employee_categories)
        embs_msg = ollama.embed(model="nomic-embed-text", input=msg)
        embs_categories = ollama.embed(model="nomic-embed-text", input=self.all_employee_categories)
        # Compute similarity scores (here we use dot product normalized by norms,
        # but you may replace this with your preferred method).
        print("processing")
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(embs_msg['embeddings']), torch.tensor(embs_categories['embeddings'])
        )
        
        print(similarities)
        # print("\n".join([f"{department}:{similarity}" for similarity, department in zip(similarities, self.all_employee_categories)]))
        
        probability = torch.max(similarities).item()
        if probability > 0.5:
            best_idx = int(torch.argmax(similarities).item())
            best_category = self.all_employee_categories[best_idx]
            print(f"Department: {best_category}")
            return best_category
        else:
            return None
        
    def llm_response(self, system_prompt: str, user_msg: str) -> str:
        # In your real implementation, this method would call your LLM.
        # For demonstration, we return a dummy response.
        print("Going to respond")
        response = ollama.generate("llama2", prompt=f"{system_prompt}\nUser:{user_msg}")
        return response['response']
    
    
    def reasoner(self, system_prompt: str, user_msg: str) -> str:
    
        print("Going to respond")
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content": user_msg
                },
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.75,
            stream=True,
            stop=None,
        )
        response = ""
        for chunk in completion:
            # print(chunk.choices[0].delta.content or "", end="")
            response += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        return response


if __name__ == "__main__":
    bot = EmployeeFinder()

    test_queries = [
        "I need a project lead for a web development project with moderate scale.",
        "Looking for an AI engineer who can manage deep learning tasks and possibly mentor junior developers.",
        "Need a DevOps engineer with experience in cloud technologies for a high-availability project.",
        "We require a QA engineer for automation testing in an agile environment.",
        "Find me a full stack developer experienced in Node.js and MongoDB who can start immediately.",
        "Need a good suggestion for someone to work on Quantum Computing research, thing is we got a new chip",
        "We are venturing into Quantum Machine Learning. We need an engineer to research and potentially develop solutions. No one in the company has direct experience, so suggest someone we can train."
    ]

    query = "We need a cybersecurity expert specialized in penetration testing and vulnerability assessment for a critical infrastructure project. If we don't have a direct match, who can we cross-train?"
    # print(f"Query: {test_queries[3]} ")
    # bot.get_relevant_category(test_queries[3])
    print("User Query:", query)
    result = bot(query)
    print("LLM Response:")
    print(result)
    print("-" * 80)

    # for query in test_queries:
    #     print("User Query:", query)
    #     result = bot(query)
    #     print("LLM Response:")
    #     print(result)
    #     print("-" * 80)
    #     break
