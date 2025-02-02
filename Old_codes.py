from groq import Groq
from sentence_transformers import SentenceTransformer
import torch

# John: (Python Developer) fresher\nCarol: (Product lead) two years in the company
employee_data = {
    "Junior web developers":"Don: (Junio Web Developer) fresher\n",
    "Senior web developers":"Ron: (Web Developer: Highly Expert) Two years in the company\n",
    "Junior AI Engineer": "Elijah Patel: fresher, with a degree in Computer Science and a focus on Machine Learning.\nAva Lee: 6 months of experience, worked on several AI projects involving natural language processing\nLiam Reed: 1 year of experience, skilled in deep learning and computer vision.",
    "Project Leads":"Sophia Rodriguez: 3 years of experience, led multiple projects involving web development and team management\nJulian Styles: 5 years of experience, expert in Agile methodologies and project planning\nMaya Jensen: 4 years of experience, strong background in software development and team leadership",
    "Project Managers":"Project Manager",
    "Marketing":"Marketing",
    "Python Developer":""
}


allocation_data = {
    "AI Projects":[],
    "Web Projects":[]
}

class RAGBot():

    def __init__(self) -> None:
        
        self.client = Groq(api_key="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")
        self.model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

        self.employee_data = employee_data

    def __call__(self, msg:str):

        emply_data = self.do_rag(msg)

        print(f"Rag Results: {emply_data}")

        completion = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an AI HR, your task is to allocate workers to projects and tasks based on there skills, expertise etc.\nNot only that you can also plan to train some workers to do a specific task for which there are no experts available at the moment.\n\nWorker Data:\n\n{emply_data}\n\nAllocation Data:\nRon is already working in a project for company websites major bug fixes\nCarol is working on a small scale work of leading the team that Ron is in\n\nOutput format should be like,\nOptimal(<name>):<description>\nSuggestion(<name>):<description>\n"
                },
                {
                    "role": "user",
                    "content": msg
                },
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        responses = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                responses = responses + chunk.choices[0].delta.content
        return responses
    
    def do_rag(self, msg:str):
        keys = [key for key in list(employee_data.keys())]
        embs1 = self.model.encode([msg])
        embs2 = self.model.encode(keys)

        similarity_scores = self.model.similarity(embs1, embs2)
        idx = torch.argmax(similarity_scores).item()


        return self.employee_data[keys[idx]]

# client = Groq(api_key="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")
if __name__ == "__main__":

    bot = RAGBot()
    prompt_text = bot("I want a project lead for a minor web development project")
    print(prompt_text, "\n\n")
    # print(bot.do_rag("I want a project lead for leading an AI team of junior devs"))
    
