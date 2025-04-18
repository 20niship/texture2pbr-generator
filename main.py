from runner import runner_main
import cv2


def main():
    res = runner_main("example.jpg")  # Replace with your image path
    cv2.imshow("normal", res["normal"])
    cv2.imshow("roughness", res["roughness"])
    cv2.imshow("occlusion", res["occlusion"])
    cv2.imshow("height", res["height"])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
