# Taken from https://huggingface.co/docs/transformers/en/model_doc/internvl
# Showing different styles of inference

# Using AutoModelForImageTextToText
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch


def inference_using_pipeline():
  from transformers import pipeline
  
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
              },
              {"type": "text", "text": "Describe this image."},
          ],
      },
  ]
  
  pipe = pipeline("image-text-to-text", model="OpenGVLab/InternVL3-1B-hf")
  outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
  outputs[0]["generated_text"]

def text_only_inference():
  torch_device = "cuda"
  model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
  processor = AutoProcessor.from_pretrained(model_checkpoint)
  model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
  
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Write a haiku"},
          ],
      }
  ]
  
  inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device, dtype=torch.bfloat16)
  
  generate_ids = model.generate(**inputs, max_new_tokens=50)
  decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
  
  print(decoded_output)


def single_image():
  torch_device = "cuda"
  model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
  processor = AutoProcessor.from_pretrained(model_checkpoint)
  model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
  
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
              {"type": "text", "text": "Please describe the image explicitly."},
          ],
      }
  ]
  
  inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
  
  generate_ids = model.generate(**inputs, max_new_tokens=50)
  decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
  
  decoded_output


def batch_image():
  torch_device = "cuda"
  model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
  processor = AutoProcessor.from_pretrained(model_checkpoint)
  model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
  
  messages = [
      [
          {
              "role": "user",
              "content": [
                  {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                  {"type": "text", "text": "Write a haiku for this image"},
              ],
          },
      ],
      [
          {
              "role": "user",
              "content": [
                  {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                  {"type": "text", "text": "Describe this image"},
              ],
          },
      ],
  ]
  
  
  inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
  
  output = model.generate(**inputs, max_new_tokens=25)
  
  decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
  decoded_outputs  

def batch_multi_image():
  torch_device = "cuda"
  model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
  processor = AutoProcessor.from_pretrained(model_checkpoint)
  model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)
  
  messages = [
      [
          {
              "role": "user",
              "content": [
                  {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                  {"type": "text", "text": "Write a haiku for this image"},
              ],
          },
      ],
      [
          {
              "role": "user",
              "content": [
                  {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
                  {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
                  {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
              ],
          },
      ],
  ]
  
  inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
  
  output = model.generate(**inputs, max_new_tokens=25)
  
  decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
  decoded_outputs
