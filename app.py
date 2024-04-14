from flask import Flask, request, flash, redirect, url_for, session, render_template, abort, jsonify
from PIL import ImageEnhance, Image
# import pytesseract
import base64
import requests
import openai
import io
import os
import json
import numpy as np

from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import load_model, load_processor
from surya.settings import settings

app = Flask(__name__)
client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/run_ocr', methods=['POST'])
def run_ocr():
    uploaded_files = request.files.getlist('images')
    images = [Image.open(io.BytesIO(file.read())) for file in uploaded_files]

    return get_ordered_text(images)

# full OCR pipeline - upload pages, improve contrast, detect layouts, run text detection, fix typos, reorder paragraphs
def get_ordered_text(images):
  
  #checks image contrast
  for i in range(len(images)):
    image = images[i]    
    estimated_contrast = sample_image_contrast(image)
    print(f"Estimated contrast ratio: {estimated_contrast:.2f}")

    # improves low contrast images
    if (estimated_contrast < 4.5):
      enhancer = ImageEnhance.Contrast(image)
      scale = 4.5 / estimated_contrast - 0.25 # enhances lower contrast images more
      images[i] = enhancer.enhance(scale)


  # surya ocr layout detection
  layout, success = layout_detection(images)
  if (success):
    print(f"Layout detection complete")
  else:
    print(f"Layout detection failed - falling back on GPT-4V")
    # OPTIONALLY, call gpt_section on each page and do not run the following code
    layout = []

  url = 'https://ai.fix.school/pytesseract' # my endpoint, hosted elsewhere to resolve dependency issues
  texts = []

  # text detection for each detected bounding box
  for ocr_result in layout:
      for text_line in ocr_result.bboxes:
          bbox = text_line.bbox # bounding box
          crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

          # optionally, you can resize the crop to improve resolution with the two commented out lines below
          # w, h = crop.size 
          # high_res_crop = crop.resize((round(w * 2), round(h * 2)), Image.LANCZOS)
          
          text = call_pytesseract(crop, url) # text detection - returns false if fails

          print(f"text: {text} \n bbox: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} \n \n")
          
          if text:
            text = fix_typos(text)
            print(f"fixed typos: {text} \n")
            texts.append(text)

  # GPT 4.5 paragraph reordering
  instructions = '''given a sequence of paragraphs (each with a letter before them), return in JSON their appropriate order. Assume they are mostly in correct sequence, but occassionally sequences are separated (eg if one paragraph says "cont. on page 6" and another says "cont from page 1" you could assume the paragraphs preceding the first indicator and following the second belong next to each other). Where you're unsure, put the paragaph that came first in the given order, first. Respond ONLY with json and include EVERY paragraph. format: { list: ["3", "4", "1" ...]}'''
  order_json = order_paragraphs(texts, instructions)
  print(order_json)
  if not is_valid_json(order_json):
    print("not valid json")
    return '\n'.join(text)

  data = json.loads(order_json)

  final_order = data['list']

  reordered_final = []
  print(texts)

  for position in final_order:
    reordered_final.append(texts[int(position) - 1])

  return "\n".join(reordered_final)

def layout_detection(images): # must pass array
  model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
  processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
  det_model = load_model()
  det_processor = load_processor()

  try:
    line_predictions = batch_text_detection(images, det_model, det_processor)
    layout_predictions = batch_layout_detection(images, model, processor, line_predictions)
    return layout_predictions, True
  except Exception as e:
    return e, False
  
def calculate_luminance(color):
    # https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
    r, g, b = color
    r, g, b = [x/255.0 for x in (r, g, b)] # normalize, then do something called gamma correction
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(l1, l2):
    # https://ux.stackexchange.com/questions/107318/formula-for-color-contrast-between-text-and-background
    return (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)

# randomly sample image to identify highest and lowest luminance areas (text and background colors) and get contrast by dividing the two
def sample_image_contrast(image, num_samples=1000, top_n=100):
    data = np.array(image)

    num_rows, num_columns, _ = data.shape

    row_indices = np.random.randint(0, num_rows, size=num_samples)
    col_indices = np.random.randint(0, num_columns, size=num_samples)
    colors = data[row_indices, col_indices]
    
    luminances = np.array([calculate_luminance(color) for color in colors])

    sorted_indices = np.argsort(luminances)
    top_indices = sorted_indices[-top_n:]  # Highest luminance
    bottom_indices = sorted_indices[:top_n]  # Lowest luminance

    top_colors = np.mean(colors[top_indices], axis=0)
    bottom_colors = np.mean(colors[bottom_indices], axis=0)

    # Calculate luminance for the averaged extreme colors
    l1 = calculate_luminance(top_colors)
    l2 = calculate_luminance(bottom_colors)

    # Calculate and return contrast ratio
    return contrast_ratio(l1, l2)

# Set up and calling external pytesseract API to resolve dependency issues
def call_pytesseract(img, url):
  buffered = io.BytesIO()
  img.save(buffered, format="JPEG")
  img_byte = buffered.getvalue()

  files = {'image': ('image.jpg', img_byte, 'image/jpeg')}

  response = requests.post(url, files=files)

  if response.status_code == 200:
      return(response.json()['text'])
  else:
      return(False)

# Pytesseract is cheaper but less effective than GPT-4V, so we go through with a second pass here
def fix_typos(paragraph):
  instructions = "you are a typo-fixing machine. you'll be given text with some clear mistakes and unnecessary line breaks. you will return ONLY the corrected text, NOTHING else. If the text appears to be gibberish, return NOTHING."
  completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages= [
                {"role": "system", "content": instructions},
                {"role": "user", "content": paragraph}
            ]
        )
  response = completion.choices[0].message.content
  clean_data = response.replace('`', '')

  return response



# GPT 4.5 paragraph reordering
def order_paragraphs(paragraphs, instructions):
  count = 1
  input = ""
  for paragraph in paragraphs:
    input += str(count) + ":" + paragraph + "\n"
    count += 1

  completion = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages= [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input}
            ]
        )
  response = completion.choices[0].message.content
  clean_data = response.replace('`', '')
  cleaner_data = clean_data.replace('json', '')
  return cleaner_data

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False

# the more expensive option for text detection
def gpt_section(image: Image.Image):
  buffered = io.BytesIO()
  image.save(buffered, format="JPEG")
  image_byte = buffered.getvalue()

  # Encode bytes to base64
  base64_image = base64.b64encode(image_byte).decode('utf-8')

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }                     
  payload = {
    "model": "gpt-4-1106-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Return all the text in this image and nothing else. If there is no text, return a blank space."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
              "detail": "low"
            },

          }
        ]
      }
    ],
    "max_tokens": 800
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return(response.json())