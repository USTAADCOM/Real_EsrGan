"""
main module to run Real-ESERGAN gradio demo
"""
import os
from random import randint
import sys
from subprocess import call
import gradio as gr
from PIL import Image
import torch
import psutil
torch.hub.download_url_to_file('http://people.csail.mit.edu/billf/project%20pages/sresCode/Markov%20Random%20Fields%20for%20Super-Resolution_files/100075_lowres.jpg', 'bear.jpg')
def run_command(command: str)-> None:
    """
    run_command module take os command as input and execute this command.

    Parameters
    ----------
    command: str
        command user want to execute as os command.
    
    Return
    ------
    None
    """
    try:
        print(command)
        call(command, shell = True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
run_command("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P .")
run_command("pip install basicsr")
run_command("pip freeze")
os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P .")
def refine_image(input_image: Image, mode: str)-> str:
    """
    refine_image method take a Pillow image and mode as input and after enhancing 
    the image resolution save it and return the image path.

    Parameters
    ----------
    input_image: Image
        Pillow image input by te user.
    mode: str
        mode of image base or anime.
    """
    random_id = randint(1, 10000)
    input_dir = "/content/Real-ESRGAN/input_image" + str(random_id) + "/"
    output_dir = "/content/Real-ESRGAN/output_image" + str(random_id) + "/"
    run_command("rm -rf " + input_dir)
    run_command("rm -rf " + output_dir)
    run_command("mkdir " + input_dir)
    run_command("mkdir " + output_dir)
    basewidth = 256
    wpercent = basewidth/float(input_image.size[0])
    hsize = int((float(input_image.size[1])*float(wpercent)))
    input_image = input_image.resize((basewidth,hsize), Image.LANCZOS)
    input_image.save(input_dir + "1.jpg", "JPEG")
    if mode == "base":
        run_command("python inference_realesrgan.py -n RealESRGAN_x4plus -i "+ input_dir + " -o " + output_dir)
    else:
        os.system("python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i "+ input_dir + " -o " + output_dir)
    return os.path.join(output_dir, "1_out.jpg")

TITLLE = "Real-ESRGAN Demo"
gr.Interface(
    refine_image,
    [gr.inputs.Image(type = "pil", label = "Input"),gr.inputs.Radio(["base","anime"], 
                                                                    type = "value", 
                                                                    default = "base", 
                                                                    label = "model type")],
    gr.outputs.Image(type = "pil", label = "Output"),
    title = TITLLE,
    examples = [
    ['bear.jpg','base'],
    ['anime.png','anime']
    ]).launch(share = True)
