from spot import Spot, HAND_RGB_UUID, image_response_to_cv2
import cv2


def main(spot: Spot):
    window_name = 'spot camera viewer'
    try:
        while True:
            # Get Spot camera image
            image_responses = spot.get_image_responses([HAND_RGB_UUID])
            hand_image_response = image_responses[0]  # only expecting one image
            img = image_response_to_cv2(hand_image_response)

            # Draw a circle "crosshair" at the center
            height, width = img.shape[:2]
            cx, cy = width // 2, height // 2
            img = cv2.circle(
                img,
                center=(cx, cy),
                radius=5,
                color=(0, 0, 255),
                thickness=1
            )

            cv2.imshow(window_name, img)
            cv2.waitKey(1)
    finally:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    spot = Spot("ViewCamera")
    main(spot)
