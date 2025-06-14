import uuid
from llama_index.embeddings.ollama import OllamaEmbedding
import torch


ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

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


class VectorDB:
    
    def __init__(self, employee_dict:dict):
        
        self.keys = list(employee_dict.keys())
        
        pass_embedding = ollama_embedding.get_text_embedding_batch(
        [employee_dict[key] for key in list(employee_dict.keys())],
        show_progress=True
        )
        # print(len(pass_embedding))
        self.pass_embedding_t = torch.tensor(pass_embedding)
        
        self.employee_dict = employee_dict
    
    def get(self, query:str, top_k=3):
    
        query_embedding = ollama_embedding.get_query_embedding(query)
        # print(len(query_embedding))

        query_embedding_t = torch.tensor(query_embedding)

        out = torch.cosine_similarity(query_embedding_t, self.pass_embedding_t)
        # print(out)
        _, indices = torch.topk(out, k=top_k)
        # print(indices)
        topk_employee_ids = [self.keys[idx] for idx in indices]
        
        return topk_employee_ids

    def get_info(self, employee_ids):
        print("Top Matches:\n") 
        result = [self.employee_dict[id] for id in employee_ids]
        formatted_output = "\n\n".join(result)
        print(formatted_output)
        # return formatted_output

def generate_employee_text_data(employee_data):
    employee_dict = {}
    
    for category, employees in employee_data.items():
        for employee in employees:
            employee_id = str(uuid.uuid4())  # Generate a unique ID for each employee
            
            details = (
                f"Name: {employee['name']}, "
                f"Role: {employee['role']}, "
                f"Department: {category}, "
                f"Skills: {', '.join(employee['skills'])}, "
                f"Experience: {employee['experience_years']} years, "
                f"Busy Level: {employee['busy_level']}, "
                f"Can Multi-task: {'Yes' if employee['can_multi_task'] else 'No'}, "
                f"Quick Learner: {'Yes' if employee['quick_learner'] else 'No'}"
            )
            
            employee_dict[employee_id] = details
    
    return employee_dict


if __name__ == "__main__":
    # Example usage
    employee_data_with_ids = generate_employee_text_data(employee_data)
    # print(employee_data_with_ids.keys())
    # print(employee_data_with_ids[list(employee_data_with_ids.keys())[0]])
    
    Db = VectorDB(employee_data_with_ids)
    # See Keyword search is required cause if there is no employee with given years of experience or skills he should be
    # marked orange in the UI
    employee_ids = Db.get("I want an AI or ML engineer with more that 5 years of experience and he should be not able to multitask") 
    Db.get_info(employee_ids)   
    
    