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

    transfVecs = withTransform(vecs, transform=fM, offset=offset)

    spn = [transfVecs[vecNames.index(x)] for x in ["idX","idY","x2y","x2z","y2x","y2z"]]
    basePlaneVecs = drawVectors(128, spn)
    basePlaneVecs.save(f"out/bpv.png")

    

    #invering matrix
    #
    invM = np.linalg.inv(fM)    
    print(fM)
    print(invM)

    rayVecs = {}
    for x in range(17):
        for y in range(17):
            rayVT = (np.array([x/16,y/16,0]), np.array([0,0,1]))
            dic = getAllPlaneCollition(
                vecs=transfVecs,
                vecNames=vecNames,
                rayVT=rayVT
            )

            rayVecs[f"{x},{y},L"] = dic["L"]
            rayVecs[f"{x},{y},R"] = dic["R"]
            rayVecs[f"{x},{y},T"] = dic["T"]

    unshiftedRays = withTransform(
        rayVecs, transform=np.identity(3), offset=-offset)

    invRays = withTransform(
        unshiftedRays, transform=invM)
    
    testInvInv = withTransform(invRays, transform=fM, offset=offset)    

    """
        #x    
        for i,vt in invRays.items():
            if vt == None: continue

            (o,v) = vt
            sc = np.array(o)
            (ox,oy,oz) = sc
            #print(i,sc)

        #for i,(o,v) in enumerate(invRays):
        #    invRays[i] = (np.array([0,0,0]),v)

        #print()
        #print(
        #    "x",np.dot(np.cross([1,0,0],[0,1,0]),
        #    np.array([0.5,0.5,1])-np.array([0,0,1])
        #))

        #print()
        #print(invRays[idxBeam(7,7)])
        """

    renderFace(
        rays=invRays,
        nPixels=16,
        renderSize=128,
        face="L"
    )
    topM = np.array([[0,0,1],[1,0,0],[0,1,0]])
    toprays = withTransform(invRays,transform=topM)
    renderFace(
        rays=toprays,
        nPixels=16,
        renderSize=128,
        face="T"
    )

    #notOOb = copy.deepcopy(invRays)
    #for x in range(17):
    #    for y in range(17):
    #        vt = invRays[idxBeam(x,y)]
    #        (o,v) = vt
    #        epsi = 1e-6
    #        if not (min(o) > -epsi and max(o) < 1+epsi):
    #            notOOb[idxBeam(x, y)] = (np.zeros(3),np.zeros(3))
"""
    pointImg = drawVectors(128, invRays)
    pointImg.save(f"out/__plane.png")

    topRotate = np.array([0.13,0.24,0])
    
    diagonals = []
    for x in range(16):
        for y in range(16):
            try:
                (ldwn,_) = invRays[idxBeam(x,y)]
                (lup,_) = invRays[idxBeam(x,y+1)]
                (rdwn,_) = invRays[idxBeam(x+1,y)]
                (rup,_) = invRays[idxBeam(x+1,y+1)]
            except:
                continue
            
            #diagonals.append((lup,rdwn-ldwn))
            diagonals.append((ldwn,rup-ldwn))
            diagonals.append((ldwn,lup-ldwn))
            diagonals.append((ldwn,rdwn-ldwn))
            for _ in range(40):
                diagonals.append((np.zeros(3),np.zeros(3)))

    pointAndDiag = diagonals + invRays

    drawVectors(128, pointAndDiag).save("out/__diag.png")

    transfPointAndDiag = withTransform(diagonals, transform=fM, offset=offset)
    drawVectors(128, transfPointAndDiag).save("out/__undiag.png")
    
    invinvPoint = withTransform(invRays, transform=fM, offset=offset)
    drawVectors(128, invinvPoint).save("out/__unpoint.png")

    return

    
    
    # Drawing

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
    """


def renderFace(rays=None, nPixels=None, renderSize=None, face=None):
    polys = []
    for y in range(nPixels):
        for x in range(nPixels):
            (p1,_) = rays[f"{x  },{y  },{face}"]
            (p2,_) = rays[f"{x+1},{y  },{face}"]
            (p3,_) = rays[f"{x+1},{y+1},{face}"]
            (p4,_) = rays[f"{x  },{y+1},{face}"]

            polys.append([p1,p2,p3,p4])

    front = drawPolys(
        size=128,
        polygons=polys
    )
    front.save(f"out/f/rend_{face}.png")

    return

#Only works for rayVT = ([_,_,_], [0,0,_])
#THIS IS WRONG, picks the wrong collition point
def getAllPlaneCollition(vecs, vecNames, rayVT):
    
    (oRay, vRay) = rayVT
    [oRayX,oRayY,oRayZ] = oRay

    lPlaneBaseVecs = [vecs[vecNames.index(x)] for x in ["idX","idY"]]
    rPlaneBaseVecs = [vecs[vecNames.index(x)] for x in ["x2y","x2z"]]
    tPlaneBaseVecs = [vecs[vecNames.index(x)] for x in ["y2x","y2z"]]
    planes = [lPlaneBaseVecs,rPlaneBaseVecs,tPlaneBaseVecs]

    cpz = {}
    for i,[(o1,v1),(o2,v2)] in enumerate(planes):
        planeName = ["L","R","T"][i]

        planeNormal = np.cross(v1, v2)
        planePoint = o1+0.5*(v1+v2)

        collitionPoint = LinePlaneCollision(
            planeNormal = planeNormal, 
            planePoint = planePoint, 
            rayDirection = vRay, 
            rayPoint = oRay
        )

        cpz[planeName] = (collitionPoint,np.zeros(3))
        
    """
        #print(collitionPoint-o1)

        #onPlane = np.dot(planeNormal, collitionPoint-o1)
        #if abs(onPlane) < 1e-6:
        #    print(planeName,oRay*16,collitionPoint[2])

        d1 = np.dot(v1,collitionPoint-o1)
        d2 = np.dot(v2,collitionPoint-o2)

        #epl = np.dot(planeNormal,collitionPoint-o1)
        
        #if(abs(epl) < 0.001):
        #    print(planeName,epl,collitionPoint)
        #    return (collitionPoint,np.zeros(3))

        projPlane = np.dot(planeNormal, collitionPoint-o1)
        onPlane = abs(projPlane) < 1e-6
        
        if (0<=d1<=1):
            if (0<=d2<=1):
                #print(planeName,collitionPoint,d1,d2)
                return (collitionPoint,np.zeros(3))
        
        #if(onPlane):
            #print(planeName,collitionPoint)"""
    return cpz

def idxBeam(x,y):
    return x+y*17

def shrink(n,fnum):
    ret = fnum
    for x in range(n):
        ret = math.nextafter(ret,-math.inf)
    return ret

def withTransform(vectup: {(np.array,np.array)}, transform=None, offset=None):
    try: 
        if(offset == None):
            offset = np.zeros(3)
    except:
        ""

    vecs = copy.deepcopy(vectup)

    itr = None
    if type(vecs) is dict:
        itr = vecs.items()
    else:
        itr = enumerate(vecs)

    for i,vt in itr:
        if vt == None: continue

        (org,v) = vt
        v = np.matmul(transform,v)
        org = np.matmul(transform,org)+offset
        vecs[i] = (org,v)
    return vecs

def drawPolys(size, polygons: [[np.array]], hue=0, img=None):
    if img == None:
        render = Image.new(
            mode="RGBA", 
            size=(size,size), 
            color=(255, 0, 0, 0))
    else:
        render = img

    draw = ImageDraw.Draw(render)
    hue = 0
    for polyp in polygons: 
        p2d = []
        for p in polyp:
            [x,y,z] = p
            p2d.append((size*x, size*y))

        draw.polygon(p2d,fill=f"hsl({hue}, 100%, 50%)")
        hue = (hue+10)%360

    return render

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

    for vt in vectors:
        if vt == None: continue

        (org,v) = vt
        width=1
        if (np.linalg.norm(v)<1e-6):
            draw.rectangle((
                    s*(org[0])-0.5,
                    s-s*(org[1])-0.5,
                    s*(org[0])+0.5,
                    s-s*(org[1])+0.5
                ),
                fill=f"hsl({x}, 100%, 50%)",
                width=1
            )
        else:
            draw.line((
                    s*org[0],
                    s-s*org[1],
                    s*(org[0]+v[0]),
                    s-s*(org[1]+v[1])
                ),
                fill=f"hsl({x}, 100%, 50%)",
                width=1
            )
        x = (x+1)%256

    return render

def printv(tup):
    try:
        if tup==None:
            return
    except:None
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

    psi = w + si * rayDirection + planePoint
    
    return psi

run()