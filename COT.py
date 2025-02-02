import json
import ollama
import torch

# --------------------------------------------------------------------
# ENHANCED EMPLOYEE DATA: Using structured dictionaries with extra fields
# --------------------------------------------------------------------
employee_data = {
    "Software Developers": [
        {
            "name": "Alice Johnson",
            "role": "Full Stack Developer",
            "skills": ["Python", "React", "Django"],
            "experience_years": 3,
            "busy_level": "low",            # low, medium, high
            "can_multi_task": True,
            "quick_learner": True
        },
        {
            "name": "Bob Smith",
            "role": "Backend Developer",
            "skills": ["Java", "Spring Boot", "SQL"],
            "experience_years": 5,
            "busy_level": "medium",
            "can_multi_task": False,
            "quick_learner": False
        },
        {
            "name": "Charlie Kim",
            "role": "Frontend Developer",
            "skills": ["HTML", "CSS", "JavaScript", "React"],
            "experience_years": 2,
            "busy_level": "high",
            "can_multi_task": True,
            "quick_learner": True
        }
    ],
    "AI/ML Engineers": [
        {
            "name": "Dana Lee",
            "role": "Data Scientist",
            "skills": ["Python", "TensorFlow", "Pandas"],
            "experience_years": 4,
            "busy_level": "medium",
            "can_multi_task": True,
            "quick_learner": True
        },
        {
            "name": "Evan Wu",
            "role": "ML Engineer",
            "skills": ["Python", "PyTorch", "scikit-learn"],
            "experience_years": 2,
            "busy_level": "low",
            "can_multi_task": True,
            "quick_learner": True
        },
        {
            "name": "Fiona Zhang",
            "role": "Research Scientist",
            "skills": ["Python", "Deep Learning", "NLP"],
            "experience_years": 6,
            "busy_level": "high",
            "can_multi_task": False,
            "quick_learner": True
        }
    ],
    "Project Managers": [
        {
            "name": "George Martin",
            "role": "Senior Project Manager",
            "skills": ["Agile", "Scrum", "Leadership"],
            "experience_years": 7,
            "busy_level": "medium",
            "can_multi_task": True,
            "quick_learner": False
        },
        {
            "name": "Hannah Davis",
            "role": "Project Manager",
            "skills": ["Kanban", "Communication", "Team Coordination"],
            "experience_years": 3,
            "busy_level": "low",
            "can_multi_task": True,
            "quick_learner": True
        }
    ],
    "QA Engineers": [
        {
            "name": "Ian Thompson",
            "role": "Automation QA",
            "skills": ["Selenium", "Python", "Testing"],
            "experience_years": 4,
            "busy_level": "medium",
            "can_multi_task": False,
            "quick_learner": True
        },
        {
            "name": "Jasmine Patel",
            "role": "Manual QA",
            "skills": ["Testing", "Bug Reporting"],
            "experience_years": 2,
            "busy_level": "low",
            "can_multi_task": True,
            "quick_learner": True
        }
    ],
    "DevOps": [
        {
            "name": "Kevin Brown",
            "role": "DevOps Engineer",
            "skills": ["AWS", "Docker", "Kubernetes"],
            "experience_years": 5,
            "busy_level": "medium",
            "can_multi_task": True,
            "quick_learner": False
        },
        {
            "name": "Liam Chen",
            "role": "Site Reliability Engineer",
            "skills": ["GCP", "Terraform", "Monitoring"],
            "experience_years": 3,
            "busy_level": "low",
            "can_multi_task": True,
            "quick_learner": True
        }
    ]
}

# This sample allocation_data might be used to flag employees already in projects,
# or record extra contextual information.
allocation_data = {
    "ongoing_projects": [
        {"name": "Alice Johnson", "project": "E-Commerce Revamp"},
        {"name": "George Martin", "project": "Internal Tool Upgrade"}
    ]
}

# --------------------------------------------------------------------
# RAGBot Class: Upgraded to work with structured data
# --------------------------------------------------------------------
class RAGBot:
    def __init__(self) -> None:
        # In your production code, you’d use your Groq (or similar) API key.
        # self.client = Groq(api_key="YOUR_API_KEY")
        # self.model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
        self.employee_data = employee_data
        self.all_employee_categories = list(self.employee_data.keys())
        print(self.all_employee_categories)

    def __call__(self, msg: str) -> str:
        # First, determine which category is most relevant using semantic similarity.
        category = self.get_relevant_category(msg)
        # Retrieve the list of employees for that category.
        candidates = self.employee_data.get(category)
        
        # Optionally, you could add filtering here based on extra criteria from the msg.
        # For now, we simply pass the candidate list to the LLM along with context.

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

        # Here we simulate a chat completion (replace with your Groq client or LLM API call).
        # For demonstration, we simply echo back the prompt with the user's message.
        response = self.fake_llm_response(system_prompt, msg)
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
        best_idx = int(torch.argmax(similarities).item())
        best_category = self.all_employee_categories[best_idx]
        print(best_category)
        return best_category

    def fake_llm_response(self, system_prompt: str, user_msg: str) -> str:
        # In your real implementation, this method would call your LLM.
        # For demonstration, we return a dummy response.
        print("Going to respond")
        response = ollama.generate("llama2", prompt=f"{system_prompt}\nUser:{user_msg}")
        return response['response']
        # response_lines = [
        #     "Optimal(Alice Johnson): A full stack developer with moderate experience, low workload, "
        #     "and strong adaptability. Suitable for handling both front-end and back-end tasks.",
        #     "Suggestion(Dana Lee): With her expertise in AI/ML and quick learning ability, she can be trained to handle "
        #     "additional responsibilities for the project if required."
        # ]
        # return "\n".join(response_lines)

# --------------------------------------------------------------------
# TESTING THE UPGRADED PLATFORM WITH DIFFERENT SCENARIOS
# --------------------------------------------------------------------
if __name__ == "__main__":
    bot = RAGBot()

    test_queries = [
        "I need a project lead for a web development project with moderate scale.",
        "Looking for an AI engineer who can manage deep learning tasks and possibly mentor junior developers.",
        "Need a DevOps engineer with experience in cloud technologies for a high-availability project.",
        "We require a QA engineer for automation testing in an agile environment."
    ]

    query = "Need a good suggestion for someone to work on Quantum Computing research, thing is we got a new chip"
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
