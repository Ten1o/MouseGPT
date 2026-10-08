import os
import json
import base64
import argparse
import requests
import time
import cv2
import numpy as np


def insert_newlines(s, interval=150):
    chunks = [s[i:i+interval] for i in range(0, len(s), interval)]
    return '\n'.join(chunks)

def put_multiline_text(img, text, position, font, font_scale, color, thickness):
    lines = text.split('\n')
    x, y = position
    for i, line in enumerate(lines):
        line_y = y + i * int(font_scale * 25 + thickness)
        img = cv2.putText(img, line, (x, line_y), font, font_scale, color, thickness)
    return img

def process_images(file1, file2, index):
    output_basename = os.path.basename(file1).replace('.jpg', '.json')
    if os.path.exists(os.path.join(output_folder, output_basename)):
        print(f'{os.path.join(output_folder, output_basename)} has been generated.')
        with open(os.path.join(output_folder, output_basename)) as f:
            return json.load(f)
    
    max_retries = 500
    retries = 0
    success = False

    while retries < max_retries and not success:
        try:
            with open(os.path.join(data1_path, file1), 'rb') as f:
                img1_base64 = base64.b64encode(f.read()).decode('utf-8')
            with open(os.path.join(data2_path, file2), 'rb') as f:
                img2_base64 = base64.b64encode(f.read()).decode('utf-8')

            v_id = str(int(os.path.basename(file1.split('.')[0])))

            prompt = "The photos taken at different views at the same time is as follows."
            prompt += "The overall movement speed of the mouse is: {:.2f} cm/s, the speed of the head: {:.2f} cm/s, speed of right and left forelimbs: {:.2f}, {:.2f} cm/s, and right and left hind limbs are {:.2f}, {:.2f} cm/s respectively.".format(
                velocity_list[v_id]["base"],
                velocity_list[v_id]["head"],
                velocity_list[v_id]["forepawL"],
                velocity_list[v_id]["forepawR"],
                velocity_list[v_id]["hindpawL"],
                velocity_list[v_id]["hindpawR"],
                )

            data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_base64}", "detail":"high"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_base64}", "detail":"high"}}
                        ]
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.5,
                "max_tokens": 2000
            }

            headers = {
                'Content-Type': 'application/json',
                'api-key': api_key
            }
            
            response = requests.post(
                f"https://{RESOURCE_NAME}.openai.azure.com/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version=2024-04-01-preview",
                headers=headers,
                json=data
            )

            with open(os.path.join(log_folder, output_basename), "w") as f:
                json.dump(data, f)
            
            if response.status_code == 200:
                result = response.json()
                print(type(result))
                result['system_prompt'] = system_prompt
                result['prompt'] = prompt
                with open(os.path.join(output_folder, output_basename), "w") as f:
                    json.dump(result, f)
                success = True
            else:
                raise Exception(f"Error: {response.status_code}, {response.text}")
        
        except Exception as e:
            if 'Your input image may contain content that is not allowed by our safety system.' in str(e):
                print(f"Content not allowed: {output_basename}")
                retries = max_retries
            else:
                print(f"Retry {retries + 1} for {output_basename}: {e}")
                retries += 1
                time.sleep(10)

    if not success:
        with open(os.path.join(output_folder, output_basename), "w") as f:
            json.dump(["Error"], f)
        print(f"Failed to process {output_basename} after {max_retries} retries.")
    return result


system_prompt = "The user will show you one frame of a photo or several photos of a mouse in a glass chamber shot at different views at the same time, and necessary information such as the absolute value of velocity, etc.. "

system_prompt += "You need to give me the description of the motion of the mice in the photos. You should give me a .json format, where there are several elements. You should describe the motion of the mouse in each part of json using full sentences as detailed as possible. You should answer each of the following questions. Your response should include but not limited to information of below questions. "

# stationary / in motion? 
system_prompt += "The first element in the json is \"Overall\". From the images, what is the overall status of the mouse, stationary or in motion? If it is moving, the movement is slow, moderate or rapid? What is this mouse doing? Is it standing, walking, or doing nothing?\n"

# face & head
system_prompt += "Next element: \"Head\". It should be the description of the head and face region. What can you see from its face? Is it just looking around? Or is it licking / grooming its paw or fur? What is the head position and orientation with respect to its body or the ground? Is the head turning left, turning right or upward? Which direction is it looking into? \n"

# limbs
system_prompt += "Next element: \"Limb\". What is its front paw / front foot doing? Is it on the ground or in the air? What is its hind paw / hind leg doing? Is it on the ground or in the air? Is it clearly walking? What is its tail doing? Is it extended on the ground, in the air parallel to the ground or pointed upwards?\n"

# body
system_prompt += "Next element: \"Torso\". What can you see from its main body, or torso? Is the spine curved vertically, horizontally, or stretched? Are the chest, abdomen and hip (or butt) partially contacted with the ground, fully lay on the ground, or in the air? \n"

# other info
system_prompt += "Next element: \"Others\". What else can you directly see about the motion or behavior of this mouse in this photo? Please include other infomation and describe it as detailed as possible, but still the description should be facts, rather than opinions or assumptions. \n"

system_prompt += "And the last element would be the \"Keywords\". You should use line of keywords separated by commas to conclude and summarize the identifiable features of the mouse behavior in each of the above components.\n"

system_prompt += "Note that your description / keywords should only be about the animal itself, without the environments. And your answers should only contain facts, without opinions or assumptions. Do NOT use negative sentence such as \"rather than\", or \"instead of\". The main part of the description should come from the image, and the speed data is only used as an auxiliary. The velocity is an absolute value without direction, so a high velocity does not necessarily mean the mouse is walking or running, it might just caused by vertical movement of the body, such as standing up. And the value of velocity is only an estimate with noise, so a low value such as 1 cm/s might be only caused by noise or shaking of the mouse, instead of the actual movement, a high value might also caused by a detection failure. Thus, the major conclusion should only come from the images themselves, and the velocity is only a cross-validation. Also, do NOT include exact number of speed in your respond, use natural language to describe it instead. \n"

    
output_folder = 'output_des'
img_output_folder = 'output_img'

log_folder = 'log'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(img_output_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)



parser = argparse.ArgumentParser(description="Generate descriptions for images using OpenAI's API.")
parser.add_argument("--data_folder", type=str, help="Base folder for data processing.")
parser.add_argument("--api", type=str, required=True, help="Base folder for data processing.")
parser.add_argument("--resource", type=str, required=True, help="The name of your Azure OpenAI resource.")
parser.add_argument("--deploy", type=str, required=True, help="The name of your GPT-4 Turbo with Vision model deployment.")
args = parser.parse_args()


api_key = args.api
RESOURCE_NAME = args.resource
DEPLOYMENT_NAME = args.deploy



with open('img_list.txt', 'r') as f:
    testcase_list = f.readlines()
testcase_list = [l.strip() for l in testcase_list]


for data in testcase_list:
    
    data_folder = os.path.dirname(os.path.dirname(os.path.dirname(data)))
    basename = os.path.basename(data)
    number = str(int(basename.split('.')[0]))
    data1_path = f'{data_folder}/pre_images/data3/{basename}'
    data2_path = f'{data_folder}/pre_images/data5/{basename}'
    
    velocity_path = f'{data_folder}/velocity.json'
    with open(velocity_path, 'r') as file:
        velocity_list = json.load(file)
    velocity = velocity_list[number]
    print(data1_path, data2_path, velocity)
    result = process_images(data1_path, data2_path, number)
    text = result["choices"][0]["message"]["content"]
    print(text)
    
    
    text = insert_newlines(text)
    
    position = (10, 10) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (0, 255, 255)
    thickness = 2
    
    img = np.hstack([cv2.imread(data1_path), cv2.imread(data2_path)])
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    img_with_text = put_multiline_text(img, text, position, font, font_scale, color, thickness)
    
    name = os.path.basename(data1_path)
    output_path = os.path.join(img_output_folder, name)
    cv2.imwrite(output_path, img_with_text)