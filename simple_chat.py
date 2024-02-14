import gradio as gr
import requests
import os

model_endpoint = os.getenv('MODEL_ENDPOINT', 'http://localhost:8080')

def ask_question(question):
    response = requests.post("%s/question" % model_endpoint, json={"question": question})
    res = response.json()
    answer = res['answer']
    time = res['time_taken']
    return answer, time

def get_model_info():
    response = requests.get("%s/info" % model_endpoint)
    info = response.json()
    return info

prompt_output = gr.Textbox("Here's my answer!", label="My Answer", lines=10)
time_output = gr.Textbox("0.0", label="Time Taken (s)", lines=1)

iface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs=[prompt_output, time_output],
    title="Chat with SLM",
    description="Type a question and get an answer from the LLM. " + str(get_model_info())
)
iface.launch(inline=True, share=False, server_name="0.0.0.0", server_port=8088)