# import ollama
# import time
# # Your input text
# input_text = "Hello, how are you today?"

# # Generate a response
# start = time.time()
# response = ollama.generate(model="llama2", prompt=input_text)
# end = time.time()
# print(f"Generated response {end-start}: {response['response']}")

# start = time.time()
# input_text = "Write a python code for prompting a message on a window on the system when executed"
# response = ollama.generate(model="llama2", prompt=input_text)
# end = time.time()
# print(f"Generated response {end-start}: {response['response']}")


import ollama

response = ollama.embed(model="nomic-embed-text", input=['Software Developers', 'AI/ML Engineers', 'Project Managers', 'QA Engineers', 'DevOps'])
print(response['embeddings'])
print(response)



# import json
# employee_data = {
#     "Junior web developers":[],
#     "Senior web developers":[],
#     "Junior AI Engineer": [],
#     "Project Leads":[],
#     "Project Managers":[],
#     "Marketing":[]
# }

# allocation_data = {
#     "AI Projects":[],
#     "Web Projects":[]
# }


# from groq import Groq

# client = Groq()
# completion = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {
#             "role": "system",
#             "content": "You are an AI HR, your task is to allocate workers to projects and tasks based on there skills, expertise etc.\nNot only that you can also plan to train some workers to do a specific task for which there are no experts available at the moment.\n\nWorker Data:\n\nRon: (Web Developer: Highly Expert) Two years in the company\nDon: (Junio Web Developer) fresher\nJohn: (Python Developer) fresher\nCarol: (Product lead) two years in the company\n\nAllocation Data:\nRon is already working in a project for company websites major bug fixes\nCarol is working on a small scale work of leading the team that Ron is in\n\nOutput format should be like,\nOptimal(<name>):<description>\nSuggestion(<name>):<description>\n"
#         },
#         {
#             "role": "user",
#             "content": "I need web developers for making a site for hospital, its just a simple one so, I think I won't need that much of expertise, suggest picks"
#         },
#         {
#             "role": "assistant",
#             "content": "Optimal(Don): Don, as a junior web developer, can handle a simple website project for a hospital, and it would be a good opportunity for him to gain experience and develop his skills.\nSuggestion(Ron): Although Ron is already working on a project, if the hospital website project is urgent and requires high-quality output, Ron's expertise as a highly experienced web developer could be utilized, but it would depend on his current workload and availability.\nSuggestion(Train John): If Don is not available or if you want to have another option, John, who is a Python developer, could be trained to work on web development for the hospital website project, as Python is often used in web development, and with some training, John could learn the necessary skills to handle a simple website project."
#         }
#     ],
#     temperature=1,
#     max_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )

# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")
