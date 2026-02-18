import cv2

from drawcv import drawcv


def main() -> None:
    image_path = "test.png"
    output_path = "test_visionframe_output.png"

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image. Check the path to '{image_path}'.")

    detections = [
        {"bbox": [450.57825, 224.44461, 520.823, 318.31363], "confidence": 1.00},
        {"bbox": [557.1738, 214.80641, 632.3633, 306.88293], "confidence": 1.00},
        {"bbox": [798.4944, 255.20078, 884.0017, 363.41306], "confidence": 1.00},
        {"bbox": [783.3083, 69.67408, 857.413, 160.07814], "confidence": 1.00},
        {"bbox": [1289.9105, 76.19713, 1387.7126, 201.35213], "confidence": 1.00},
        {"bbox": [1081.1265, 244.8385, 1181.2657, 368.23102], "confidence": 1.00},
        {"bbox": [597.03064, 355.4427, 688.59576, 460.47357], "confidence": 1.00},
    ]

    styles = [
        "pro-clean-blue",
        "pro-slate",
        "pro-minimal-black",
        "futuristic-hud",
        "futuristic-circuit",
        "funny-confetti",
        "pro-gold"
    ]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        style_name = styles[i % len(styles)]

        drawcv(
            image=image,
            style_id=style_name,
            coords=(x1, y1, x2, y2),
        )

        cv2.putText(
            image,
            f"{style_name} | conf {det['confidence']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")

    cv2.imshow("visionframe Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

