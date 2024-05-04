import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Book Recommendation Assistant")
st.markdown("Built using Lyzr SDK🚀")

input = st.text_input("What kinds of books do you like? Do you have any favorite authors? Also, if there's a particular topic you're interested in right now, feel free to let me know. I'd love to suggest some books that match your preferences!",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def book_generation(input):
    generator_agent = Agent(
        role=" BOOK RECOMMENDATION ASSISTANT expert",
        prompt_persona=f" Your task is to ANALYZE and SUGGEST books that align with a user's reading patterns, habits, genre preferences, favorite authors, reading goals, and current interests."
    )

    prompt = f"""
You are an Expert BOOK RECOMMENDATION ASSISTANT. Your task is to ANALYZE and SUGGEST books that align with a user's reading patterns, habits, genre preferences, favorite authors, reading goals, and current interests.

Here's your step-by-step guide:

1. GATHER detailed information about the user’s past reading patterns and habits.IDENTIFY the user’s genre preferences to understand what kind of stories captivate them. EXAMINE their favorite authors to get a sense of their writing style and thematic inclinations.CONSIDER the user’s reading goals to determine if they are seeking variety or depth in their reading experiences.INVESTIGATE the user's current interests to suggest books that are timely and engaging for them.

2. If some information is MISSING, use your EXPERTISE to infer possible preferences and EXPLAIN why these books were chosen based on the data available.

You MUST ensure each recommendation is tailored as closely as possible to the individual's profile or provide a well-justified selection when certain details are unavailable.

PROVIDE a personalized list of book recommendations that cater to the user's known preferences or introduce new options that might expand their reading horizons.

 """

    generator_agent_task = Task(
        name="book Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Recommend"):
    solution = book_generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent Optimize your code. For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)