from typing import TypedDict, Literal, cast
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
from langgraph.func import entrypoint, task
from langchain_google_genai import ChatGoogleGenerativeAI


_: bool = load_dotenv(find_dotenv())

api_key = "AIzaSyDX47VMsoVe-nie5pqzIbmB3qLlGciHUBs"

# Initialize different model instances
router_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21", api_key=api_key)
simple_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", api_key=api_key)
advanced_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", api_key=api_key)
reasoning_coding_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp-01-21", api_key=api_key)


class InputState(TypedDict):
    task: str


TaskType = Literal["write", "research", "post", "unknown"]


class TaskClassifier(BaseModel):
    type: TaskType


@task
def classify_task(task: str) -> TaskType:
    """Classify the task based on its nature: 
    - 'write': For tasks that involve content creation or writing
    - 'research': For tasks that require gathering information or data
    - 'post': For tasks that involve sharing or posting content
    """
    prompt = """Analyze the following task and classify it as either:
    - 'write': For tasks that involve content creation or writing
    - 'research': For tasks that require gathering information or data
    - 'post': For tasks that involve sharing or posting content
    
    
    If you don't able to classify it, then respond with only word as "unknown"!
    
    Task: {task}
    """

    response = cast(TaskClassifier, advanced_model.with_structured_output(
        TaskClassifier).invoke(prompt.format(task=task)))
    result = response.type

    if result not in ["write", "research", "post"]:
        return "unknown"  # Default to complex if classification is unclear
    return result  # type: ignore


@task
def handle_writing_task(task: str) -> str:
    """Handle writing tasks for a digital marketer agent."""
    prompt = f"""As a digital marketer, please create a compelling piece of content for the following task: {
        task}"""
    response = simple_model.invoke(prompt)
    return response.content


@task
def handle_research_task(task: str) -> str:
    """Handle research tasks for a digital marketer agent."""
    prompt = f"""As a digital marketer, please gather relevant information and insights for the following research task: {
        task}"""
    response = simple_model.invoke(prompt)
    return response.content


@task
def handle_posting_task(task: str) -> str:
    """Handle posting tasks for a digital marketer agent."""
    prompt = f"""As a digital marketer, please create a post for the following task: {
        task}"""
    response = advanced_model.invoke(prompt)
    return response.content


@entrypoint()
def run_workflow(input: InputState):
    """Route the task to the appropriate handler based on classification."""
    task = input.get("task", "")
    task_type = classify_task(task).result()
    print(task_type, 'task_type')

    if task_type == "write":
        answer = handle_writing_task(task).result()
    elif task_type == "research":
        answer = handle_research_task(task).result()
    elif task_type == "post":
        answer = handle_posting_task(task).result()
    else:
        answer = "Unknown task type."

    return {
        "task_type": task_type,
        "answer": answer
    }


def main():
    result = run_workflow.invoke(
        input={"task": "Write a blog post about AI Agents!"})
    print("\n\n", "Generated task: ", result)


main()
