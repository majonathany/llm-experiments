import json, os, time, torch
from threading import Thread
from faker import Faker
from datetime import datetime
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextIteratorStreamer
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

model_name_or_path = "/home/ubuntu/llm_experiments/Yarn-Mistral-7B-128k-AWQ"
# model_name_or_path = "TheBloke/Yarn-Mistral-7B-128k-AWQ"
# model_name_or_path = "berkeley-nest/Starling-LM-7B-alpha"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
# Load model

model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True)


class SpecialTextIteratorStreamer(TextIteratorStreamer):
    filename = ""



async def generate_inference(filename,  temperature = None, top_p=None, top_k=None, max_new_tokens = None, do_sample = None, token_ratio = None, repetition_penalty=None, stream_message=None):

    global model, tokenizer

    json_settings = get_settings()

    temperature_global = json_settings.get('temperature')
    top_p_global = json_settings.get('top_p')
    top_k_global = json_settings.get('top_k')
    max_new_tokens_global = json_settings.get('max_new_tokens')
    do_sample_global = json_settings.get('do_sample')
    token_ratio_global = json_settings.get('token_ratio')
    repetition_penalty_global = json_settings.get('repetition_penalty')


    if not temperature:
        temperature = temperature_global

    if not top_p:
        top_p = top_p_global

    if not top_k:
        top_k = top_k_global

    if not max_new_tokens:
        max_new_tokens = max_new_tokens_global

    if not repetition_penalty:
        repetition_penalty = repetition_penalty_global

    context = open(f'/home/ubuntu/llm_experiments/input/{filename}', 'r').read().strip()

    prompt_template=f'{context}'
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    output_filename = filename.split('.')[0] + f"-{formatted_datetime}-output." + filename.split('.')[1]


    token_input = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    input_token_len = len(token_input[0])

    if max_new_tokens is None:
        max_new_tokens = int(len(token_input[0]) * token_ratio)

    start_time = time.time()

    print(f"*** Running generate at {current_datetime} with params: repetition_penalty {repetition_penalty} temperature = {temperature}, top_p={top_p}, top_k={top_k}, max_new_tokens = {max_new_tokens}, do_sample = {do_sample}")

    streamer = SpecialTextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    if True:

        kwargs = dict(
                  do_sample=True,  # Enabling stochastic mode
                  temperature=temperature,  # Controls randomness
                  top_p=top_p,  # Nucleus sampling parameter
                  top_k=top_k,  # Top-k sampling parameter
                  max_new_tokens=max_new_tokens,  # Limit on token generation
                  repetition_penalty=float(repetition_penalty),
                  streamer=streamer)
    else:
        kwargs = dict(
          do_sample=False,  # Disabling stochastic mode
          max_new_tokens=max_new_tokens,  # Limit on token generation
          repetition_penalty=float(repetition_penalty),
          streamer=streamer)

    print(kwargs, token_input)
    thread = Thread(target=model.generate, args=[token_input], kwargs=kwargs)

    thread.start()
    full_message = []
    generated_text = ""
    for i, new_text in enumerate(streamer):
        if stream_message:
            print(new_text)
            await stream_message.stream_token(new_text)
            if i == 0:
                full_message.extend(tokenizer.encode(new_text)[:-1])
            else:
                full_message.extend(tokenizer.encode(new_text)[1:-1])

    end_time = time.time()

    print(f"Elapsed time was {end_time-start_time} seconds")

    # Get the tokens from the output, decode them, print them

    generation_output = full_message
    print(generation_output)
    text_output = tokenizer.decode(generation_output)
    generation_output_len = len(generation_output)
    token_output_len = len(generation_output)
    output = f"{text_output}\n:\nLengths: {input_token_len}, {generation_output_len} {token_output_len}\n\n================\nLLM output ({formatted_datetime} temperature = {temperature}, top_p={top_p}, top_k={top_k}, repetition_penalty = {repetition_penalty} max_new_tokens = {max_new_tokens}, do_sample = {do_sample}\n\n\n\n\n==================\nOriginal Input\n\n {prompt_template}"

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    output = f"{text_output}\n:\nLengths: {input_token_len}, {generation_output_len} {token_output_len}\n\n================\nLLM output ({formatted_datetime} temperature = {temperature}, top_p={top_p}, top_k={top_k}, repetition_penalty = {repetition_penalty} max_new_tokens = {max_new_tokens}, do_sample = {do_sample}\n\n\n\n\n==================\nOriginal Input\n\n {prompt_template}"

    write_output(f'/home/ubuntu/llm_experiments/output/{output_filename}', output)
    write_output(f'/home/ubuntu/llm_experiments/output/output.txt', output)




    return text_output

def write_output(output_filename, text):
    with open(output_filename, 'w+') as output_file:
        output_file.write(text)

def get_settings():
    settings_filename = '/home/ubuntu/llm_experiments/llm_settings.json'

    temperature = 0.3
    top_p=0.95
    top_k=40
    max_new_tokens = 1024
    do_sample = True
    token_ratio = 1.5
    repetition_penalty=1.1

    if os.path.isfile(settings_filename):
        settings_file = open(settings_filename, 'r')
        json_settings = json.load(settings_file)

        temperature = json_settings.get('temperature')
        top_p = json_settings.get('top_p')
        top_k = json_settings.get('top_k')
        max_new_tokens = json_settings.get('max_new_tokens')
        do_sample = json_settings.get('do_sample')
        token_ratio = json_settings.get('token_ratio')
        repetition_penalty = json_settings.get('repetition_penalty')

        settings_file.close()


    temperature = temperature if temperature is not None else 0.3
    top_p = top_p if top_p is not None else 0.95
    top_k = top_k if top_k is not None else 40
    max_new_tokens = max_new_tokens if max_new_tokens is not None else 1024
    do_sample = do_sample if do_sample is not None else True
    token_ratio = token_ratio if token_ratio is not None else 1.5
    repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.1

    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "token_ratio": token_ratio,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

@cl.on_chat_start
async def start():
    json_settings = get_settings()

    print(json_settings)

    temperature = json_settings.get('temperature')
    top_p = json_settings.get('top_p')
    top_k = json_settings.get('top_k')
    max_new_tokens = json_settings.get('max_new_tokens')
    do_sample = json_settings.get('do_sample')
    token_ratio = json_settings.get('token_ratio')
    repetition_penalty = json_settings.get('repetition_penalty')

    if not temperature:
        temperature = 0.3
    if not top_p:
        top_p = .95
    if not top_k:
        top_k = 40
    if not max_new_tokens:
        max_new_tokens = 1024
    if not repetition_penalty:
        repetition_penalty = 1.1

    settings = await cl.ChatSettings(
        [
            Slider(
                id="temperature",
                label="Temperature for LLM",
                initial=temperature,
                min=0,
                max=2,
                step=0.05,
            ),
            Slider(
                id="top_k",
                label="top_k for inference",
                initial=top_k,
                min=1,
                max=50,
                step=1,
            ),
            Slider(
                id="top_p",
                label="top_p for inference",
                initial=top_p,
                min=0,
                max=1,
                step=0.001,
            ),
            Slider(
                id="max_new_tokens",
                label="Max number of new tokens to generate",
                initial=max_new_tokens,
                min=1,
                max=4096,
                step=1,
            ),
            Slider(
                id="repetition_penalty",
                label="Repetition Penalty",
                initial=repetition_penalty,
                min=0,
                max=3,
                step=0.05,
            )
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    temperature = settings.get('temperature')
    top_p = settings.get('top_p')
    top_k = settings.get('top_k')
    max_new_tokens = settings.get('max_new_tokens')
    token_ratio = settings.get('token_ratio')
    repetition_penalty = settings.get('repetition_penalty')
    do_sample = settings.get('do_sample')

    new_json = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "token_ratio": token_ratio,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    settings_filename = '/home/ubuntu/llm_experiments/llm_settings.json'
    settings_file = open(settings_filename, 'w+')
    json.dump(new_json, settings_file)
    settings_file.close()

    print("on_settings_update", settings)
    # if settings['Temperature']:
    #     temperature_global = settings['Temperature']
    # if settings['top_k']:
    #     top_k_global = settings['top_k']
    # if settings['top_p']:
    #     top_p_global = settings['top_p']
    # if settings['max_new_tokens']:
    #     max_new_tokens_global = settings['max_new_tokens']
    # if settings['repetition_penalty']:
    #     repetition_penalty_global = settings['repetition_penalty']

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"{current_datetime}.txt"

    newfile = open(f'/home/ubuntu/llm_experiments/input/{filename}', 'w+')
    newfile.write(message.content)
    newfile.close()

    # back = open(f'{output_filename}', 'r')
    # contents = back.read()
    #
    msg = cl.Message(content="")

    #
    fake = Faker()
    prompt = fake.text()
    #
    try:
        contents = await generate_inference(filename=filename, stream_message = msg)
    except Exception as e:
        raise e
    finally:
        torch.cuda.empty_cache()



async def return_words(words):
    for word in words:
        yield word
        time.sleep(0.1)

from fastapi import FastAPI, JSONResponse
from pydantic import BaseModel


from chainlit.server import app

@app.post("/api/prompt")
def hello(request: Request):
    print(request.headers)
    prompt = request.data

    return JSONResponse(request.data)
