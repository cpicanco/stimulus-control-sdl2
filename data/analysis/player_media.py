import os
import pickle

import cv2
from PIL import (Image, ImageFont, ImageDraw)
import numpy as np

pictures = {}
raleway_150 = None
raleway = None
picanco = None
arimo   = None
default_font_color = (255, 255, 255)

monitor_width = 1440
instruction_width = int((monitor_width//3)*2)

instructions = {}

def save_pictures(participant_id, override=False):
    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    pictures_filename = 'pictures.pkl'
    rel_path = os.path.join('media', 'png', participant_id)
    base_path = os.path.join(base_dir, rel_path)
    pictures_path = os.path.join(base_path, pictures_filename)
    # check if file exists
    if os.path.exists(pictures_path) and not override:
        return

    base_path_assets = os.path.join(base_dir, 'media', 'assets')

    filenames = []
    filenames.append('AudioPicture.png')
    for filename in os.listdir(base_path):
        filenames.append(filename)

    pictures = {}
    for filename in filenames:
        if 'AudioPicture' in filename:
            filepath = os.path.join(base_path_assets, filename)
        else:
            filepath = os.path.join(base_path, filename)

        if filename.endswith('.png'):
            key = filename.replace('.png', '')
            if filename == 'AudioPicture.png':
                value = cv2.imread(filepath)
                value[value == 255] = 0
            else:
                value = cv2.imread(filepath)
            value = cv2.resize(value, (208, 208), interpolation=cv2.INTER_AREA)
            pictures[key] = value

    with open(pictures_path, 'wb') as f:
        pickle.dump(pictures, f)
        print(f'Pictures saved to {pictures_path}')

def load_pictures(participant_id):
    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    rel_path = os.path.join('media', 'png', participant_id)
    base_path = os.path.join(base_dir, rel_path)

    pictures_path = os.path.join(base_path, 'pictures.pkl')

    if os.path.exists(pictures_path):
        with open(pictures_path, 'rb') as f:
            pictures = pickle.load(f)
    else:
        raise Exception(f'Pictures not found: {pictures_path}')

    return pictures

def load_fonts():
    global raleway
    global picanco
    global arimo
    global raleway_150

    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    rel_path = os.path.join('media', 'fonts')
    base_path = os.path.join(base_dir, rel_path)

    for filename in os.listdir(base_path):
        filepath = os.path.join(base_path, filename)
        if filename == 'Raleway-Regular.ttf':
            raleway = ImageFont.truetype(filepath, 50)
            raleway_150 = ImageFont.truetype(filepath, 150)

        elif filename == 'Picanco_et_al.ttf':
            picanco = ImageFont.truetype(filepath, 230)

        elif filename == 'Arimo-Regular.ttf':
            arimo = ImageFont.truetype(filepath, 230)


def get_wrapped_text(text: str, font: ImageFont.ImageFont, line_length: int):
    lines = ['']
    for word in text.split():
        line = f'{lines[-1]} {word}'.strip()
        if font.getlength(line) <= line_length:
            lines[-1] = line
        else:
            lines.append(word)
    return lines

def save_words():
    from words import all
    load_fonts()

    words_ = {}
    for word in all:
        if word == 'bolo' or word == 'bala':
            width = 435
            height = 257
        elif word == 'Fim.':
            width = 287
            height = 177
        else:
            width = 514
            height = 230
        buffer = np.zeros((height, width, 3), dtype=np.uint8)
        buffer = draw_word(buffer, 0, 0, word)
        words_[word] = buffer

    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    rel_path = os.path.join('media', 'fonts')
    base_path = os.path.join(base_dir, rel_path)
    words_path = os.path.join(base_path, 'words.pkl')
    # save dictionary
    with open(words_path, 'wb') as f:
        pickle.dump(words_, f)

def load_words():
    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    rel_path = os.path.join('media', 'fonts')
    base_path = os.path.join(base_dir, rel_path)
    words_path = os.path.join(base_path, 'words.pkl')

    with open(words_path, 'rb') as f:
        words = pickle.load(f)

    return words

def save_instructions():
    load_fonts()
    txt_instructions = {}
    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    rel_path = os.path.join('media', 'instructions')
    base_path = os.path.join(base_dir, rel_path)

    for filename in os.listdir(base_path):
        filepath = os.path.join(base_path, filename)
        if filename.endswith('.txt'):
            key = filename.replace('.txt', '')
            value = open(filepath, 'r', encoding='utf-8').read()
            txt_instructions[key] = value

    for key, text in txt_instructions.items():
        lines = get_wrapped_text(text, raleway, instruction_width)
        _, _, _, line_height = raleway.getbbox(lines[0])
        height = len(lines) * line_height
        buffer = np.zeros((height, instruction_width, 3), dtype=np.uint8)
        pil_image = Image.fromarray(buffer)
        canvas = ImageDraw.Draw(pil_image)
        offset = 0
        for line in lines:
            canvas.text((0, offset), line, font=raleway, fill=default_font_color)
            offset += line_height

        instructions[key] = np.array(pil_image)

    instructions_path = os.path.join(base_path, 'instructions.pkl')
    # save dictionary
    with open(instructions_path, 'wb') as f:
        pickle.dump(instructions, f)

def load_instructions():
    base_dir = os.path.dirname(__file__)

    while not os.path.exists(os.path.join(base_dir, 'data')):
        base_dir = os.path.dirname(base_dir)

    rel_path = os.path.join('media', 'instructions')
    base_path = os.path.join(base_dir, rel_path)

    instructions_path = os.path.join(base_path, 'instructions.pkl')
    with open(instructions_path, 'rb') as f:
        instructions = pickle.load(f)

    return instructions

def draw_word(buffer, x, y, text):
    pil_image = Image.fromarray(buffer)
    canvas = ImageDraw.Draw(pil_image)
    if text == 'bolo' or text == 'bala':
        canvas.text((x, y), text, font=arimo, fill=default_font_color)
    elif text == 'Fim.':
        canvas.text((x, y), text, font=raleway_150, fill=default_font_color)
    else:
        canvas.text((x, y), text, font=picanco, fill=default_font_color)
    return np.array(pil_image)

if __name__ == '__main__':
    from fileutils import list_data_folders, data_dir

    data_dir()
    folders = list_data_folders()
    for folder in folders:
        save_pictures(folder)
    # pictures = load_pictures('1-JOP')
    # for key, value in pictures.items():
    #     cv2.imshow(key, value)
    #     cv2.waitKey(-1)
    #     cv2.destroyAllWindows()

    # save_instructions()
    # instructions = load_instructions()
    # for key, value in instructions.items():
    #     cv2.imshow(key, value)
    #     cv2.waitKey(-1)
    #     cv2.destroyAllWindows()

    # save_words()
    # words = load_words()
    # for key, value in words.items():
    #     cv2.imshow(key, value)
    #     cv2.waitKey(-1)
    #     cv2.destroyAllWindows()