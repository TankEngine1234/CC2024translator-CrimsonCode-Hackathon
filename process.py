import visual

def main(file):
    if file.filename.find('video') != -1:
        visual.process_video(file.filename)
    else:
        visual.process_image(file.filename)
    return file
