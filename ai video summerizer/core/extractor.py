#Actionableitems , decision , questions 

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 


def get_llm():
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
        )



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert executive meeting analyst and corporate assistant. "
        "Carefully review the meeting transcript and extract all actionable tasks.\n\n"
        "For each action item, use this exact format:\n"
        "1. **Task:** [Clear description of the task]\n"
        "   - **Owner:** [Person responsible, or 'Unassigned']\n"
        "   - **Deadline:** [Due date/time if stated, or 'Not specified']\n\n"
        "Strict Rules:\n"
        "- Base your response *only* on facts directly stated in the transcript.\n"
        "- Do not guess or fabricate owners or deadlines.\n"
        "- If no action items are found, output exactly: 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert executive meeting analyst and corporate assistant. "
        "Carefully review the meeting transcript and extract all finalized decisions made by the team.\n\n"
        "Format your response as a numbered list where each item highlights the core choice or agreement reached:\n"
        "1. **Decision:** [Clear summary of what was decided]\n\n"
        "Strict Rules:\n"
        "- Only include choices that were explicitly agreed upon or finalized.\n"
        "- If no key decisions were made, output exactly: 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert executive meeting analyst and corporate assistant. "
        "Carefully review the meeting transcript and extract all unresolved questions, open loops, or topics explicitly left pending for follow-up.\n\n"
        "Format your response as a numbered list:\n"
        "1. **Open Item:** [Description of the unresolved question or topic]\n"
        "   - **Raised By:** [Who asked it, if mentioned]\n\n"
        "Strict Rules:\n"
        "- Only include items that were left unanswered or explicitly marked for later.\n"
        "- If none found, output exactly: 'No open questions found.'"
    )
    return chain.invoke(transcript)