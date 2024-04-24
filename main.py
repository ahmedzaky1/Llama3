#from langchain_community.llms import Ollama
#from crewai_tools import SerperDevTool
import os
from crewai import Agent, Task, Crew, Process

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-8b-8192'  # Adjust based on available model
os.environ["OPENAI_API_KEY"] ='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'

email = "Good morning guys, an important meeting on Tuesday at 10 pm"
is_verbose = True

classifier = Agent(
    role = "email classifier",
    goal = "accuracy classify emails based on their importance. give every email one of these rating : important, casual, or spam",
    backstory = "You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be afraid to give emails bad ratings if they are not important. Your job is to help the user manage their inbox and",
    verbose = True,
    allow_delegation = False,
)

responder = Agent(
    role = "email classifier",
    goal = "Based on the email, write concise and simple responses. If the email is rated 'important' write a formal response, if email is rated 'casual' write a casual response, and if email is rated 'spam' ignore the email. no matter what, be very cocnise.",
    backstory = "You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be afraid to give emails bad ratings if they are not important. Your job is to help the user manage their inbox and",
    verbose = True,
    allow_delegation = False,
)

classify_email = Task(
    description = f"Classify the email '{email}'",
    agent = classifier,
    expected_output = "One of these three options: 'important', 'casual', or 'spam'",
)

respond_to_email = Task(
    description = f"Respond to the email: '{email}' based on the importance provided by the 'classifier' agent.",
    agent = responder,
    expected_output = "a very concise response to the email based on the importance provided by the 'classifier' agent.",
)

crew = Crew(
    agents = [classifier, responder],
    tasks = [classify_email, respond_to_email],
    verbose =2,
    Process = Process.sequential,
)

output = crew.kickoff()
print(output)