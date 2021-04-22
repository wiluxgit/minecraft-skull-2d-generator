from PIL import Image,ImageDraw
import colorsys
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

def run():
    srcFile = Image.open("assets/creeper.png")

    left=   (np.array([0,0,0]), np.array([0,1,0]))
    bottom= (np.array([0,0,0]), np.array([1,0,0]))
    right=  (np.array([1,0,0]), np.array([0,1,0]))
    top=    (np.array([0,1,0]), np.array([1,0,0]))

    orgvecs = [left,bottom,right,top]
    vecs = copy.deepcopy(orgvecs)

    r = R.from_euler('y', 45, degrees=True)
    r = R.from_euler('x', -30, degrees=True)*r

    print(r.as_matrix())


    mxL = np.array([
        np.xyR
        [ 24/64, 0    ],
        [-12/64, 28/64]
    ])
    trL = np.array([
        1/2 - 24/64,
        16.5/64
    ])

    mxR = np.array([
        [ 24/64, 0    ],
        [ 12/64, 28/64]
    ])
    trR = np.array([
        32/64,
        16.5/64-12/64
    ])

    mxT = np.array([
        [ 24/64, 24/64],
        [-12/64, 12/64]
    ])
    trT = np.array([
        1/2 - 24/64,
        16.5/64+28/64
    ])


    leftBox = withTransform(vecs, transform=mxL, offset=trL)
    vecDraw = drawVectors(64, leftBox)
    
    rightBox = withTransform(vecs, transform=mxR, offset=trR)
    vecDraw = drawVectors(64, rightBox, img=vecDraw, hue=40)

    topBox = withTransform(vecs, transform=mxT, offset=trT)
    vecDraw = drawVectors(64, topBox, img=vecDraw, hue=80)


    vecDraw.save("vec.png")

def withTransform(vectup: [(np.array,np.array)], transform=None, offset=None):
    vecs = copy.deepcopy(vectup)
    for i,(org,v) in enumerate(vecs):
        v = np.matmul(transform,v)
        org = np.matmul(transform,org)+offset
        vecs[i] = (org,v)
    return vecs

def drawVectors(size, vectors: [(np.array,np.array)], hue=0, img=None):
    if img == None:
        render = Image.new(
            mode="RGBA", 
            size=(size,size), 
            color=(255, 0, 0, 0))
    else:
        render = img
        
    draw = ImageDraw.Draw(render)

    s = size
    x = hue

    for (org,v) in vectors:
        print(org,v)

        draw.line((
                s*org[0],
                s-s*org[1],
                s*(org[0]+v[0]),
                s-s*(org[1]+v[1])
            ),
            fill=f"hsl({x}, 100%, 50%)",
            width=1
        )
        x = (x+6)%100

    return render

def printv(tup):
    try:
        (org,v) = tup
        print("o:",org,"v:",v)
    except:
        print("[")
        for x in tup:
            printv(x)
        print("]")

def makeSkullRender(inptex: Image) -> Image:
    skullRender = Image.new(
        mode="RGBA", 
        size=(64,64), 
        color=(255, 0, 0, 0))

    sideTex = inptex.crop((0,8,8,16))
    upTex = inptex.crop((8,0,16,8))
    frontTex = inptex.crop((8,8,16,16))

    draw = ImageDraw.Draw(skullRender)
    #skullRender.show()
    
    return draw

run()