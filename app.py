from dotenv import load_dotenv
import streamlit as st
import replicate
import os
import pandas as pd
import plotly.graph_objects as go

load_dotenv()
# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    replicate_api = os.getenv('REPLICATE_API_TOKEN')
    if replicate_api:
        st.success('API key already provided!', icon='✅')
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "Act as a prompt modifier. Interpret the 'User' prompt and modify it to drive meaningful, artistic input into the image generation process. Expand the user prompt creatively. Give me the modified prompt for image generation. Not include any other letter but just return the modified prompt as a response."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":0.1, "top_p":0.7, "max_length":128, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
modified_prompt = ''

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            modified_prompt = ''
            for item in response:
                modified_prompt += item

            print(modified_prompt)
                # Run the cjwbw/point-e model (text2pointcloud)
            model = replicate.models.get("cjwbw/point-e")
            version = model.versions.get("1a4da7adf0bc84cd786c1df41c02db3097d899f5c159f5fd5814a11117bdf02b")

            prediction = replicate.predictions.create(
                version=version,
                input={"prompt": modified_prompt, "output_format": "json_file"}
            )

            prediction.wait()
            
            print(prediction.output.keys())

            data = prediction.output["json_file"]

            st.success("Point cloud generated! Fetching data…")


            # ——— Parse point cloud JSON ———
            coords = data["coords"]     # list of [x, y, z]
            colors = data["colors"]     # list of [r, g, b] (0–255)

            df = pd.DataFrame(coords, columns=["x", "y", "z"])
            df[["r", "g", "b"]] = pd.DataFrame(colors)

            # Normalize colors for Plotly (0–1)
            df[["r", "g", "b"]] = df[["r", "g", "b"]] / 255.0

            # ——— Plotly 3D Scatter ———
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=df["x"],
                        y=df["y"],
                        z=df["z"],
                        mode="markers",
                        marker=dict(
                            size=2,
                            color=df[["r", "g", "b"]].values,
                            opacity=0.8,
                        ),
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)


    message = {"role": "assistant", "content": modified_prompt}
    st.session_state.messages.append(message)
