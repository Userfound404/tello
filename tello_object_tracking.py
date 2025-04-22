from utils.telloconnect import TelloConnect
from utils.followobject import FollowObject
import signal
import cv2
import argparse
import os
import shutil
import random
import time
from PIL import Image
import torch
# from realesrgan import RealESRGAN

# ========== SETUP PATHS ========== #
CAPTURE_DIR = "SwinIR/captured_frames"
ENHANCED_DIR = "enhanced_frames"
SAVE_INTERVAL = 30  # Save every ~30 frames
MAX_SAVED_FRAMES = 10  # Max number of frames to enhance

# ========== CLEAN FOLDERS ========== #
def prepare_folders():
    for folder in [CAPTURE_DIR, ENHANCED_DIR]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

# ========== IMAGE ENHANCEMENT ========== #
def enhance_images_with_realesrgan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/realesr-general-x4v3.pth')

    for img_file in os.listdir(CAPTURE_DIR):
        img_path = os.path.join(CAPTURE_DIR, img_file)
        output_path = os.path.join(ENHANCED_DIR, img_file)
        with Image.open(img_path).convert("RGB") as image:
            sr_image = model.predict(image)
            sr_image.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tello AI-enhanced tracker.')
    parser.add_argument('-model', type=str, default='')
    parser.add_argument('-proto', type=str, default='')
    parser.add_argument('-obj', type=str, default='Face')
    parser.add_argument('-dconf', type=float, default=0.7)
    parser.add_argument('-debug', type=bool, default=False)
    parser.add_argument('-video', type=str, default="")
    parser.add_argument('-vsize', type=list, default=(640, 480))
    parser.add_argument('-th', type=bool, default=False)
    parser.add_argument('-tv', type=bool, default=True)
    parser.add_argument('-td', type=bool, default=True)
    parser.add_argument('-tr', type=bool, default=True)

    args = parser.parse_args()

    pspeed = 1
    writevideo = False
    prepare_folders()
    saved_frames = 0
    frame_count = 0

    def signal_handler(sig, frame):
        raise Exception

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    tello = TelloConnect(DEBUG=args.debug, VIDEO_SOURCE=args.video if args.debug else None)
    tello.set_image_size(args.vsize)
    videow = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, tuple(args.vsize))
    if tello.debug: pspeed = 30

    tello.add_periodic_event('wifi?', 40, 'Wifi')
    tello.wait_till_connected()
    tello.start_communication()
    tello.start_video()

    fobj = FollowObject(tello, MODEL=args.model, PROTO=args.proto, CONFIDENCE=args.dconf, DEBUG=args.debug, DETECT=args.obj)
    fobj.set_tracking(HORIZONTAL=args.th, VERTICAL=args.tv, DISTANCE=args.td, ROTATION=args.tr)

    while True:
        try:
            img = tello.get_frame()
            if img is None: continue
            imghud = img.copy()
            fobj.set_image_to_process(img)

            # Frame Saving Logic
            frame_count += 1
            if frame_count % SAVE_INTERVAL == 0 and saved_frames < MAX_SAVED_FRAMES:
                fname = f"frame_{time.time():.0f}.jpg"
                cv2.imwrite(os.path.join(CAPTURE_DIR, fname), img)
                saved_frames += 1

            k = cv2.waitKey(pspeed)

        except Exception:
            tello.stop_video()
            tello.stop_communication()
            break

        fobj.draw_detections(imghud, ANONIMUS=False)
        cv2.imshow("TelloCamera", imghud)

        if k == ord('v'):
            writevideo = not writevideo

        if writevideo:
            videow.write(img)

        if k == ord('q'):
            tello.stop_communication()
            break

        if k == ord('t'):
            tello.send_cmd('takeoff')

        if k == ord('l'):
            tello.send_cmd('land')

        if k == ord('w'):
            tello.send_cmd('up 20')

        if k == ord('s'):
            tello.send_cmd('down 20')

        if k == ord('a'):
            tello.send_cmd('cw 20')

        if k == ord('d'):
            tello.send_cmd('ccw 20')

    cv2.destroyAllWindows()

    print("Enhancing saved frames using Real-ESRGAN...")
    enhance_images_with_realesrgan()
    print("Enhanced images saved to:", ENHANCED_DIR)
