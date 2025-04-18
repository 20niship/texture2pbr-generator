from runner import runner_main
import cv2
import sys

img_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"
make_seamless = True  # set input texture to seamless using img2texture

import os

if not os.path.exists(img_path):
    print(f"Image file '{img_path}' not found.")
    sys.exit(1)

if __name__ == "__main__":
    res = runner_main(img_path, make_seamless)  # Replace with your image path
    cv2.imshow("normal", res["normal"])
    cv2.imshow("roughness", res["roughness"])
    cv2.imshow("occlusion", res["occlusion"])
    cv2.imshow("height", res["height"])
    if make_seamless:
        cv2.imshow("seamless basecolor", res["basecolor"])
    cv2.waitKey(0)
