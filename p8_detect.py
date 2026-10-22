"""Phase 8 real-image check: at what pixel density (px/m) do classical face and person detectors
still find the subject, compared with the 250 px/m identification level? Phone photos, one subject."""
import cv2, numpy as np, json
from PIL import Image, ImageOps
FACE_M = 0.20   # ASSUMPTION: Haar face box height ~0.20 m (brow-chin)
files = ["IMG_2082.jpeg", "IMG_2085.jpeg", "IMG_2088.jpeg", "IMG_2089.jpeg"]
fr = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
pr = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
def load(f):
    im = ImageOps.exif_transpose(Image.open("/mnt/user-data/uploads/" + f)).convert("RGB")
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)  # 2856x2142 working copy
    return np.ascontiguousarray(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
def faces(g):
    out = []
    for c in (fr, pr):
        for flip in (False, True):
            gg = cv2.flip(g, 1) if flip else g
            r = c.detectMultiScale(gg, 1.1, 4, minSize=(14, 14))
            for (x, y, w, h) in (r if len(r) else []):
                if flip: x = gg.shape[1] - x - w
                out.append((x, y, w, h))
            if c is fr: break   # frontal: no flip needed
    return out
def persons(img):
    img = np.ascontiguousarray(img)
    if img.shape[0] < 160 or img.shape[1] < 96: return []
    r, w = hog.detectMultiScale(img, winStride=(8, 8), padding=(16, 16), scale=1.05)
    return [tuple(b) for b, s in zip(r, np.ravel(w)) if s > 0.3] if len(r) else []
res = {}
for f in files:
    img = load(f); H, W = img.shape[:2]
    # reference face at moderate size
    s0 = 1400 / H; small = cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA)
    fs = faces(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    if not fs: res[f] = "no reference face"; continue
    x, y, w, h = max(fs, key=lambda b: b[2] * b[3])
    face_h_full = h / s0; pxm_full = face_h_full / FACE_M
    cx, cy = (x + w / 2) / s0, (y + h / 2) / s0      # face centre in full-res coords
    rows = []
    for target in [400, 300, 250, 200, 160, 130, 110, 90, 75, 60, 50, 40, 32, 25, 20, 15, 10]:
        s = target / pxm_full
        if s > 1: continue
        im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hit = lambda boxes, m=1.0: any(bx/m <= cx*s <= (bx+bw)/m and by/m <= cy*s + (0 if bh < 1 else 0) <= (by+bh)/m + 0.9*H*s for (bx, by, bw, bh) in boxes)
        f_hit = any(bx <= cx*s <= bx+bw and by <= cy*s <= by+bh for (bx, by, bw, bh) in faces(g))
        p_boxes = persons(im)
        p_hit = any(bx <= cx*s <= bx+bw and by <= cy*s + 0.3*target <= by+bh for (bx, by, bw, bh) in p_boxes)
        im2 = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        p2 = persons(im2)
        p_hit2 = any(bx <= 2*cx*s <= bx+bw and by <= 2*(cy*s + 0.3*target) <= by+bh for (bx, by, bw, bh) in p2)
        rows.append(dict(pxm=target, face_px=round(target*FACE_M,1), face=f_hit, person=p_hit, person_up2=p_hit2))
    res[f] = dict(face_box_full_px=round(face_h_full), pxm_full=round(pxm_full), rows=rows)
json.dump(res, open("p8_detect.json", "w"), indent=1)
for f, r in res.items():
    if isinstance(r, str): print(f, r); continue
    print(f"\n{f}: full-res face ~{r['face_box_full_px']} px -> ~{r['pxm_full']} px/m")
    print(" px/m  face_px  face  person  person(2x up)")
    for x in r["rows"]:
        print(f" {x['pxm']:4d}  {x['face_px']:6.1f}   {'Y' if x['face'] else '.'}     {'Y' if x['person'] else '.'}       {'Y' if x['person_up2'] else '.'}")
