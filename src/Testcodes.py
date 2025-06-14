import json
import ollama
import torch
from Employee_data import employee_data, allocation_data
from Prompt import COT_prompt
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def reasoner(system_prompt: str, user_msg: str) -> str:
    
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
    
    s = json.dumps(employee_data)
    
    prompt = f"""I want an employee who can be a good candidate for the role Quantum computing engineer\n
    The Employees in the Company are:{json.dumps(employee_data)}\n
    The Employee Allocation Data is here:{json.dumps(allocation_data)}\n
    lets think step by step,
    """
    # print(s)
    # Just to look at prompt, me stuff LoL
    with open("Testpromptlook.txt", 'w') as f:
        f.write(prompt)
    
    output = reasoner(COT_prompt, prompt)
    
    print(output)
    