import cv2
import pytesseract

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to obtain binary image
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def detect_data_display_areas(image):
    # Perform image processing
    processed_image = preprocess_image(image)

    # Example: Use contours to detect rectangular regions of interest (ROI)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Example: Identify and draw rectangles around critical data areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def extract_numeric_values(image):
    # Perform image processing
    processed_image = preprocess_image(image)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Use PyTesseract to perform OCR on the image
    text = pytesseract.image_to_string(processed_image)

    return text

# Example usage
if __name__ == "__main__":
    # Load the video feed or image
    video_capture = cv2.VideoCapture("your_video_feed.mp4")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # Task 1: Detect Data Display Areas
        areas_detected_image = detect_data_display_areas(frame)

        # Task 2: Extract Numeric Values using OCR
        numeric_values = extract_numeric_values(frame)

        # Display the results
        cv2.imshow("Detected Areas", areas_detected_image)
        print("Extracted Numeric Values:", numeric_values)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
