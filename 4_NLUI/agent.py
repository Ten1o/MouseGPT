import os
import gradio as gr
from gradio import ChatMessage
from glob import glob
import uuid
import json
from datetime import datetime
import re
import pandas as pd

from runtime_config import (
    WORKSPACE_DIR, available_sessions, configure_tracing,
    image_files,
    session_config, validate_frame_ids,
)
from mousegpt_config import load_config

configure_tracing()

from tools import tool_cluster, tool_behavior, tool_search, tool_outlier, tool_classification
from langchain_experimental.tools import PythonREPLTool
import tools

gpt_tools = [tool_cluster, PythonREPLTool(), tool_behavior, tool_search, tool_outlier, tool_classification]

from langchain_openai import AzureChatOpenAI, ChatOpenAI


from langchain.agents import create_tool_calling_agent


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import AgentExecutor

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler


instructions = """You are a research assistant specializing in mouse behavior analysis. Your next task is to assist me in analyzing a mouse video using the available tools.

You have access to a Python REPL for executing code. When writing code, begin by reading the necessary files from the database, followed by reviewing the table headers to confirm the structure and content. This will help you avoid issues like key errors.

If you encounter any errors, debug the code and retry until successful.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent_config = load_config("AGENT")
if agent_config.provider == "azure_openai":
    chat_model = AzureChatOpenAI(
        azure_endpoint=agent_config.endpoint,
        azure_deployment=agent_config.model,
        api_version=agent_config.api_version,
        api_key=agent_config.api_key,
        # Some Azure deployments only accept the model default.  Older
        # langchain-openai releases otherwise serialize their 0.7 default.
        temperature=1,
    )
else:
    chat_model = ChatOpenAI(
        model=agent_config.model,
        api_key=agent_config.api_key,
        **({"temperature": 0} if agent_config.provider == "openai" else {}),
        **({"base_url": agent_config.endpoint} if agent_config.provider != "openai" else {}),
    )

agent = create_tool_calling_agent(chat_model, gpt_tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=gpt_tools, 
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=50
).with_config(
    {"run_name": "Agent"}
)


log_dir = str(WORKSPACE_DIR / "user_logs")


os.makedirs(log_dir, exist_ok=True)

def get_h5_files():
    return available_sessions()


def load_session_config(name):
    selected = session_config(name)
    with pd.HDFStore(selected["file_path"], mode="r") as store:
        descriptions = store.get("descriptions")
        features = store.get("features_des")
    selected["frame_ids"] = validate_frame_ids(descriptions["id"], features.index)
    if not image_files(selected):
        raise ValueError("No preview image matches the stored frame IDs.")
    return selected


def update_tool_config(selected_file, user_tool_config):
    selected = load_session_config(selected_file)
    selected["workspace"] = user_tool_config["workspace"]
    images = image_files(selected)
    return (
        selected,
        gr.Slider(minimum=0, maximum=len(images)-1, step=1, value=0),
        str(images[0]),
        images[0].stem,
    )


def image_selector(image_index, user_tool_config):
    images = image_files(user_tool_config)
    image_index = int(image_index)
    if not 0 <= image_index < len(images):
        raise gr.Error("Select a valid image index for this session.")
    selected_image = images[image_index]
    return str(selected_image), selected_image.stem

async def MGPT_handler(input_text, chat_history, user_id, config):
    
    
    mgpt_chat_history = []
    for message in chat_history:
        if message["role"] == "user":
            mgpt_chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            mgpt_chat_history.append(AIMessage(content=message["content"]))

    chat_history.append(ChatMessage(role="user", content=input_text))
    yield chat_history
    async for chunk in agent_executor.astream(
        {
            "input": input_text,
            "chat_history": mgpt_chat_history
        },
    ):
        if "steps" in chunk:
            for step in chunk["steps"]:
                chat_history.append(ChatMessage(role="assistant", content=str(step.action.log) + '\n' + str(step.observation),
                                  metadata={"title": f"🛠️ Used tool {step.action.tool}"}))
                
                yield chat_history
        if "output" in chunk:
            chat_history.append(ChatMessage(role="assistant", content=chunk["output"]))
            yield chat_history


async def input_handler(input_text, chat_history, user_id, file_name_display, config):
    tools.tool_config = config
    input_text = f"current frame index: {file_name_display}.\n\n" + input_text
    
    user_log_dir = os.path.join(log_dir, user_id)
    os.makedirs(user_log_dir, exist_ok=True)

    file_list = []
    for root, dirs, files in os.walk(user_log_dir):
        for file in files:
            file_list.append(os.path.join(root, file))

    async for mgpt_chat_history in MGPT_handler(input_text, chat_history, user_id, config):
        chat_history = mgpt_chat_history
        yield chat_history, "", file_list


    chat_record = [str(message) for message in chat_history]

    with open(os.path.join(user_log_dir, f"log.jsonl"), "a") as log_file:
        log_file.write(json.dumps(chat_record) + "\n")



if __name__ == "__main__":

    h5_files = get_h5_files()
    if not h5_files:
        raise SystemExit(
            "No complete example session found in 4_NLUI/data. "
            "Download the NLUI sample with the repository's sample-data downloader first."
        )
    tool_config = load_session_config(h5_files[0])

    with gr.Blocks() as demo:
        user_id = datetime.now().strftime("%y%m%d_%H%M%S_") + uuid.uuid4().hex
        

        user_tool_config = tool_config.copy()
        user_tool_config["workspace"] = os.path.join(log_dir, user_id)
        os.makedirs(user_tool_config["workspace"], exist_ok=True)

        gr.Markdown(f"## User ID: {user_id}")
        image_list = [str(path) for path in image_files(user_tool_config)]

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    type="messages",
                    label="Agent",
                    height=700
                )
                user_input = gr.Textbox(label="User Input", lines=1)
            with gr.Column(scale=1):
                dropdown = gr.Dropdown(label="Choose Data File", choices=h5_files, value=h5_files[0])
                image_output = gr.Image(label="Image", value=image_list[0])
                image_index = gr.Slider(minimum=0, maximum=len(image_list)-1, step=1, label="Choose Image", value=0)
                file_name_display = gr.Textbox(label="Image Index", value=os.path.splitext(os.path.basename(image_list[0]))[0], interactive=False)
                file_display = gr.Files(label="Log Files")

        chat_history = gr.State([])
        user_id_state = gr.State(user_id)
        config_state = gr.State(user_tool_config)


        dropdown.change(update_tool_config, inputs=[dropdown, config_state], outputs=[config_state, image_index, image_output, file_name_display])


        user_input.submit(input_handler, inputs=[user_input, chatbot, user_id_state, file_name_display, config_state], outputs=[chatbot, user_input, file_display])
        image_index.change(image_selector, inputs=[image_index, config_state], outputs=[image_output, file_name_display])

    # Tool configuration is shared in this research demo; serialize requests.
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
