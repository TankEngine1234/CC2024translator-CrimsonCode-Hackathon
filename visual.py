import os
import cv2
import shutil
import glob
import sys
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from google.cloud import translate_v2 as translate
from tesserocr import PyTessBaseAPI, PSM, OEM
import csv
import numpy as np
from skimage.util import img_as_ubyte
from skimage.morphology import disk
from skimage.filters import rank

api = PyTessBaseAPI(path="tessdata-4.1.0",lang="jpn+eng",psm=PSM.AUTO_OSD,oem=OEM.LSTM_ONLY)

def init(image_name):
    image = preprocess_image(image_name)
    api.SetImage(image)
    api.Recognize()

def get_orientation():
    it = api.AnalyseLayout()
    orientation, direction, order, deskew_angle = it.Orientation()
    return format(orientation), format(deskew_angle)

def write_TSV():
    with open("results.tsv", "w") as result:
        result.write(api.GetTSVText(0))
        result.close()

    #with open("results.tsv", "r") as result:
    #    print(result.read())

def get_regions():
    regions = []
    with open("results.tsv", "r") as tsv:
        f = csv.reader(tsv, delimiter='\t', quotechar='"')
        i = 0
        str = ""
        left = []
        top = []
        right = []
        bottom = []
        for row in f:
            if float(row[2]) == i:
                if float(row[10]) == 0 or float(row[10]) > 80:
                    str = " ".join((str, row[11]))
                    left.append(float(row[6]))
                    top.append(float(row[7]))
                    right.append(float(row[6]) + float(row[8]))
                    bottom.append(float(row[7]) + float(row[9]))
            else:
                if (not str.isspace() and str.strip()):
                    regions.append((str, min(left, default=-1), min(top, default=-1), max(right, default=-1), max(bottom, default=-1)))
                left = []
                top = []
                right = []
                bottom = []
                str = ""
                i = i + 1
        if (not str.isspace() and str.strip()):
            regions.append((str, min(left, default=-1), min(top, default=-1), max(right, default=-1), max(bottom, default=-1)))
    return regions

def end():
    api.End()

def split_video_to_png(video_path):
    output_folder = "split_images"
    os.makedirs(output_folder, exist_ok=True)
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Read the first frame
    success, frame = video_capture.read()
    count = 0

    # Loop through the video frames
    while success:
        # Write the frame as a PNG image
        cv2.imwrite(f"{output_folder}/frame_{count:04d}.png", frame)

        # Read the next frame
        success, frame = video_capture.read()
        count += 1

    # Release the video capture object
    video_capture.release()

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    img = img_as_ubyte(image)
    img_tmp = rank.equalize(np.array(img), disk(30))
    img_eq = Image.fromarray(img_tmp)
    enhancer = ImageEnhance.Contrast(img_eq)
    img_eq = enhancer.enhance(2)
    img_eq.save("temp.png")
    return img_eq

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/buoyant-song-414621-671aac12546a.json'

# Function to translate text using Google Cloud Translation API
def translate_text(text):
    client = translate.Client()
    return client.translate(text, 'en')['translatedText']

def translate_regions(regions):
    translated_region = []
    for phrase in regions:
        #remove all non standard characters
        translated_text = translate_text(phrase[0].strip())

        if any(char.isalpha() for char in translated_text):
          translated_region.append((translated_text,) + phrase[1:])
    return translated_region

# Finding the maximum font size that still fits within the box
def find_max_font_size(text, box_height, box_width):
    low = 1
    high = 1000
    max_font_size = 1

    while low <= high:
        mid = (low + high) // 2
        font = ImageFont.truetype("arial.ttf", mid)
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width <= box_width and text_height <= box_height:
            max_font_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return max_font_size

# Adding the text to the images
def add_text_to_image(image_name, text, x, y, x2, y2, is_video):
    output_folder = "translated_images/"
    os.makedirs(output_folder, exist_ok=True)

    width = x2 - x
    height = y2 - y

    if is_video: # is video we save to an additional folder
      # Check if the translated image exists
      translated_image_name = output_folder + image_name.rsplit('.', 1)[0] + "_translated." + image_name.rsplit('.', 1)[1]
      print(image_name + " - ", end="")

      if os.path.exists(translated_image_name):
        img_to_open = translated_image_name
        print("Old image")
      else:
        img_to_open = "split_images/" + image_name
        print("New image")

    else: # image so save to same dir
      translated_image_name = image_name.rsplit('.', 1)[0] + "_translated." + image_name.rsplit('.', 1)[1]
      print(image_name + " - ", end="")

      # Check if the translated file exists
      if os.path.exists(translated_image_name):
          img_to_open = translated_image_name
          print("Old image")
      else:
          img_to_open = image_name
          print("New image")

    # Open the image
    with Image.open(img_to_open) as img:
        # Create a new image in memory
        new_img = img.copy()

        # Calculate the average color of the specified box
        box = (x, y, x + width, y + height)
        box_area = new_img.crop(box)
        avg_color = tuple(map(lambda x: int(sum(x) / len(x)), zip(*box_area.getdata())))

        # Create a draw object to add text to the image
        draw = ImageDraw.Draw(new_img)

        # Load a font
        font_size = find_max_font_size(text, height, width) # Adjust font size based on the box size and text length
        font = ImageFont.truetype("arial.ttf", font_size)

        # Draw a rectangle with the average color
        draw.rectangle(box, fill=avg_color)

        # Add text on top of the rectangle
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) // 2  # Center the text within the box
        text_y = y + (height - text_height) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)

        # Save the new image with "_translated" added to the original name
        new_img.save(translated_image_name)

# Adding the text to the images
def process_image(image_name, text, x, y, x2, y2, is_video):
    output_folder = "translated_images/"
    os.makedirs(output_folder, exist_ok=True)

    width = x2 - x
    height = y2 - y

    if is_video: # is video we save to an additional folder
      # Check if the translated image exists
      translated_image_name = output_folder + image_name.rsplit('.', 1)[0] + "_translated." + image_name.rsplit('.', 1)[1]
      print(image_name + " - ", end="")

      if os.path.exists(translated_image_name):
        img_to_open = translated_image_name
        print("Old image")
      else:
        img_to_open = "split_images/" + image_name
        print("New image")

    else: # image so save to same dir
      translated_image_name = image_name.rsplit('.', 1)[0] + "_translated." + image_name.rsplit('.', 1)[1]
      print(image_name + " - ", end="")

      # Check if the translated file exists
      if os.path.exists(translated_image_name):
          img_to_open = translated_image_name
          print("Old image")
      else:
          img_to_open = image_name
          print("New image")

    # Open the image
    with Image.open(img_to_open) as img:
        # Create a new image in memory
        new_img = img.copy()

        # Calculate the average color of the specified box
        box = (x, y, x + width, y + height)
        box_area = new_img.crop(box)
        avg_color = tuple(map(lambda x: int(sum(x) / len(x)), zip(*box_area.getdata())))

        # Create a draw object to add text to the image
        draw = ImageDraw.Draw(new_img)

        # Load a font
        font_size = find_max_font_size(text, height, width) # Adjust font size based on the box size and text length
        font = ImageFont.truetype("arial.ttf", font_size)

        # Draw a rectangle with the average color
        draw.rectangle(box, fill=avg_color)

        # Add text on top of the rectangle
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) // 2  # Center the text within the box
        text_y = y + (height - text_height) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)

        # Save the new image with "_translated" added to the original name
        new_img.save(translated_image_name)

def sync_images():
    split_images_folder = "split_images"
    translated_images_folder = "translated_images"

    # Ensure the translated_images_folder exists
    os.makedirs(translated_images_folder, exist_ok=True)

    # Count the number of images in split_images_folder
    num_images = len([file for file in os.listdir(split_images_folder) if file.endswith('.png')])

    # Iterate through the images in split_images_folder
    for file in os.listdir(split_images_folder):
        if file.endswith('.png'):
            frame_number = file.split('_')[1].split('.')[0]  # Extract the frame number
            translated_file = f"frame_{frame_number}_translated.png"
            translated_path = os.path.join(translated_images_folder, translated_file)

            # Check if the translated file exists and if the frame number is lower than num_images
            if not os.path.exists(translated_path) and int(frame_number) < num_images:
                source_path = os.path.join(split_images_folder, file)
                shutil.copy(source_path, translated_path)
                print(f"Copied {file} to {translated_path}")

def images_to_video(video_name):
    imgFolder = "translated_images"
    vidName = "video_output.mp4"

    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    images = glob.glob(os.path.join(imgFolder, "*.png"))
    images.sort()

    width, height = Image.open(images[0]).size
    size = (width, height)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    if len(sys.argv) < 2:
        video = cv2.VideoWriter(vidName, codec, fps, (width, height))
    else:
        video = cv2.VideoWriter(vidName, codec, fps, size)

    for img in images:
        frame = cv2.imread(img)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

def process_image(image_name):
    # process images with ocr
    init(image_name)
    write_TSV()
    regions = get_regions()

    # translate text
    translated_regions = translate_regions(regions)

    # overlay new text on images
    for phrase in translated_regions:
        print("Image: " + image_name + " text: " + phrase[0] + " x: " + str(phrase[1]) + " y: " + str(phrase[2]) + " height: " + str(phrase[3]) + " width: " + str(phrase[4]))
        process_image(image_name, phrase[0], phrase[1], phrase[2], phrase[3], phrase[4], False)

def process_video(video_name):
    split_images_folder = 'split_images'

    # process video to images
    split_video_to_png(video_name)

    # process images with ocr
    for frame in os.listdir(split_images_folder):
      file_name = os.path.basename(frame)
      init(frame)
      write_TSV()
      regions = get_regions()

      # translate text
      translated_regions = translate_regions(regions)

      # overlay new text on images
      for phrase in translated_regions:
        print("Frame: " + frame + " text: " + phrase[0] + " x: " + str(phrase[1]) + " y: " + str(phrase[2]) + " height: " + str(phrase[3]) + " width: " + str(phrase[4]))
        add_text_to_image(frame, phrase[0], phrase[1], phrase[2], phrase[3], phrase[4], True)

    # save as video (without sound)
    sync_images()
    images_to_video(video_name)

