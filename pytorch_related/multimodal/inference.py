# Taken from https://huggingface.co/docs/transformers/en/model_doc/internvl
# Showing different styles of inference
# Also siglip2: https://huggingface.co/docs/transformers/main/model_doc/siglip2

# Using AutoModelForImageTextToText
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from transformers import pipeline


def inference_using_pipeline():  
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


def pipeline_0_shot_class():
  image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
  candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
  pipeline = pipeline(task="zero-shot-image-classification", model="google/siglip2-base-patch16-224", device=0, dtype=torch.bfloat16)
  pipeline(image, candidate_labels=candidate_labels)  


def using_siglip_class():
  # pip install -U flash-attn --no-build-isolation
  # instead of automodel use siglipmodel
  from transformers import SiglipModel
  model = SiglipModel.from_pretrained(
      "google/siglip2-so400m-patch14-384",
      attn_implementation="flash_attention_2",
      dtype=torch.float16,
      device_map=device,
  )
  # OR another way
  from transformers import Siglip2Config, Siglip2Model
  # Initializing a Siglip2Config with google/siglip2-base-patch16-224 style configuration
  configuration = Siglip2Config()
  # Initializing a Siglip2Model (with random weights) from the google/siglip2-base-patch16-224 style configuration
  model = Siglip2Model(configuration)
  # Accessing the model configuration
  configuration = model.config
  # We can also initialize a Siglip2Config from a Siglip2TextConfig and a Siglip2VisionConfig
  from transformers import Siglip2TextConfig, Siglip2VisionConfig
  # Initializing a Siglip2Text and Siglip2Vision configuration
  config_text = Siglip2TextConfig()
  config_vision = Siglip2VisionConfig()
  config = Siglip2Config.from_text_vision_configs(config_text, config_vision)


def using_autotokenizer():
  from transformers import AutoTokenizer, AutoModel
  model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
  tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
  # important: make sure to set padding="max_length" as that's how the model was trained
  inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
  with torch.no_grad():
      text_features = model.get_text_features(**inputs)  

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




def quantize_siglip2():
  from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig  
  bnb_config = BitsAndBytesConfig(load_in_4bit=True)
  model = AutoModel.from_pretrained("google/siglip2-large-patch16-512", quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa")
  processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

  url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
  image = Image.open(requests.get(url, stream=True).raw)
  candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
  # follows the pipeline prompt template to get same results
  texts = [f'This is a photo of {label}.' for label in candidate_labels]  
  # IMPORTANT: we pass `padding=max_length` and `max_length=64` since the model was trained with this
  inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt").to(model.device)
  with torch.no_grad():
      outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = torch.sigmoid(logits_per_image)
  print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")

