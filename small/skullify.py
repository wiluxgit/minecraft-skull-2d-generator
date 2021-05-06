import sys
import json
import os.path
import colorsys
from PIL import Image,ImageDraw
import numpy as np


def run():
    if len(sys.argv) > 1:
        imgpath = sys.argv[1]
    else:
        imgpath = "assets/test/tree.png"
    inpImg = Image.open(imgpath)
    if inpImg.size != (16,16):
        raise Exception("invalid size")

    f = open("skintable.json", "r")
    skinHueLocs = {}
    for k,v in json.load(f).items():
        skinHueLocs[int(k)] = v
    f.close()

    f = open("rendertable.json", "r")
    renderHueLocs = {}
    for k,v in json.load(f).items():
        renderHueLocs[int(k)] = v
    f.close()

    output = img = Image.new( 'RGBA', (1024,1024), (255, 0, 0, 0))
    pixels = output.load()

    for skinHue,outputList in skinHueLocs.items():
        try:
            renderpos = renderHueLocs[skinHue]
        except:
            raise Exception(f"{outputList[0]} is not mapped")

        (px,py) = renderpos[0]
        loc = (px,py)

        desiredcol = inpImg.getpixel(loc)
        (r,g,b,a) = desiredcol
        (h,s,v) = colorsys.rgb_to_hsv(r/255.,g/255.,b/255.)

        faceBrightnessModifier = {
            "L": 0.686,
            "F": 0.396,
            "R": 0.443,
            "B": 0.988,
            "U": 0.914,
            "D": 0 #not actually 0 but not visible
        }

        for x,y in outputList:
            face = getFace(x,y)
            vModifier = faceBrightnessModifier[face]
            nv = v/vModifier
            if nv > 1.001:
                raise Exception(f"(x,y)={loc} is too bright, max V for this location is {vModifier*100}% in HSV")
            nv = min(nv,1)
            (nr,ng,nb) = colorsys.hsv_to_rgb(h, s, nv)

            requiredColor = (int(nr*256),int(ng*256),int(nb*256),a)

            pixels[x,y] = requiredColor

    fname = os.path.basename(imgpath)
    try:
        os.mkdir("out")
    except:""
    output.save(f"out/{fname}")

def getFace(x,y):
    s = 128

    if 4*s<=x<5*s and s<=y<s*2:
        return "L"
    elif 5*s<=x<6*s and s<=y<s*2:
        return "F"
    elif 6*s<=x<7*s and s<=y<s*2:
        return "R"
    elif 7*s<=x<8*s and s<=y<s*2:
        return "B"
    elif 5*s<=x<6*s and 0<=y<s:
        return "U"
    elif 6*s<=x<7*s and 0<=y<s:
        return "D"
    else:
        raise Exception(f"(x,y)={(x,y)} is not mapped to a face, how did you even manage this?")

run()