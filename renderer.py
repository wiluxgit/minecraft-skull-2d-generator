from PIL import Image,ImageDraw
import colorsys
import numpy as np
import copy
import math
from scipy.spatial.transform import Rotation as R

def run():
    srcFile = Image.open("assets/creeper.png")

    idX = (np.array([0,0,0]), np.array([1,0,0]))
    idY = (np.array([0,0,0]), np.array([0,1,0]))
    idZ = (np.array([0,0,0]), np.array([0,0,1]))

    x2y = (np.array([1,0,0]), np.array([0,1,0]))
    x2z = (np.array([1,0,0]), np.array([0,0,1]))

    y2x = (np.array([0,1,0]), np.array([1,0,0]))
    y2z = (np.array([0,1,0]), np.array([0,0,1]))

    z2x = (np.array([0,0,1]), np.array([1,0,0]))
    z2y = (np.array([0,0,1]), np.array([0,1,0]))

    xy2z = (np.array([1,1,0]), np.array([0,0,1]))
    xz2y = (np.array([1,0,1]), np.array([0,1,0]))
    yz2x = (np.array([0,1,1]), np.array([1,0,0]))

    #orgvecs = [idX,idY,idZ,x2y,x2z,y2x,y2z,z2x,z2y,xy2z,xz2y,yz2x]
    orgvecs = [idX,idY,x2y,x2z,y2x,y2z,xy2z,xz2y,yz2x]
    vecNames = "idX,idY,x2y,x2z,y2x,y2z,xy2z,xz2y,yz2x".split(",")

    vecs = copy.deepcopy(orgvecs)

    r = R.from_euler('y', 45, degrees=True)
    r = R.from_euler('x', -30, degrees=True)*r

    rotM = r.as_matrix()
    scaleM = np.identity(3)*(1/math.sqrt(2)*3/4)
    fM = scaleM @ rotM

    offset = np.array([0.13,0.24,0]) #trial and errored

    invM = np.linalg.inv(fM)
    beamVecs = [0 for _ in range(17*17)]
    for x in range(17):
        for y in range(17):
            beamVecs[x+y*17] = (np.array([x,y,-10]), np.array([0,0,100]))
            

    
    print(fM)
    print(invM)

    transfVecs = withTransform(vecs, transform=fM, offset=offset)

    pixels = 16*4
    vecDraw = drawVectors(pixels, transfVecs)
    
    #rightBox = withTransform(vecs, transform=mxR, offset=trR)
    #vecDraw = drawVectors(64, rightBox, img=vecDraw, hue=40)

    #topBox = withTransform(vecs, transform=mxT, offset=trT)
    #vecDraw = drawVectors(64, topBox, img=vecDraw, hue=80)

    imgZoom = (64*3)//pixels
    
    shouldRender = Image.open(f"assets/white_head_mc_{pixels}.png")
    shouldRender.resize((pixels*imgZoom,pixels*imgZoom),Image.NEAREST).save(f"out/_vanilla{pixels}.png")

    prend = np.asarray(vecDraw)
    vrend = np.asarray(shouldRender)
    diff = vrend+prend
    diffDraw = Image.fromarray(diff)

    diffDraw = diffDraw.resize((pixels*imgZoom,pixels*imgZoom),Image.NEAREST)
    diffDraw.save(f"out/diff{pixels}.png")
    
    vecDraw.save(f"out/unscaled{pixels}.png")
    vecDraw = vecDraw.resize((pixels*imgZoom,pixels*imgZoom),Image.NEAREST)
    vecDraw.save(f"out/render{pixels}.png")

def getBestPlaneCollition(vecs, vecNames):
    lPlaneBaseVecs = [vecs[vecNames.index(x)] for x in ["idX","idY"]]
    rPlaneBaseVecs = [vecs[vecNames.index(x)] for x in ["x2y","x2z"]]
    tPlaneBaseVecs = [vecs[vecNames.index(x)] for x in ["y2x","y2z"]]

def shrink(n,fnum):
    ret = fnum
    for x in range(n):
        ret = math.nextafter(ret,-math.inf)
    return ret

def withTransform(vectup: [(np.array,np.array)], transform=None, offset=None):
    try: 
        if(offset == None):
            offset = np.array([0,0,0])
    except:
        ""

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
    x += 10

    for (org,v) in vectors:
        draw.line((
                s*org[0],
                s-s*org[1],
                s*(org[0]+v[0]),
                s-s*(org[1]+v[1])
            ),
            fill=f"hsl({x}, 100%, 50%)",
            width=1
        )
        x = (x+33)%256

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

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
	    raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

run()