import cv2
import numpy as np


#Color Defining
colors = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Eraser": (0, 0, 0)
}

#Default colour
current_color = colors["Blue"]

#Default brush thickness
brush_thickness = 5

#Webcam 
cap = cv2.VideoCapture(0)

width = int(cap.get(3))
height = int(cap.get(4))

#Blank canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

#Previous coordinates
prev_x, prev_y = 0, 0

#Minimum contour area
min_area = 1000

#HSV range for green marker
lower_green = np.array([40, 100, 100])
upper_green = np.array([90, 255, 255])

#Morphological kernel
kernel = np.ones((5, 5), np.uint8)


#Main Loop
while True:

    ret, frame = cap.read()

    if not ret:
        break

    #Flip frame
    frame = cv2.flip(frame, 1)

    #Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    #Noise reduction
    mask = cv2.dilate(mask, kernel, iterations=1)

    #Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    #Draw UI Buttons
    cv2.rectangle(frame, (10, 10), (110, 60), (122, 122, 122), -1)
    cv2.putText(frame, "CLEAR", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.rectangle(frame, (130, 10), (210, 60), colors["Blue"], -1)
    cv2.rectangle(frame, (230, 10), (310, 60), colors["Green"], -1)
    cv2.rectangle(frame, (330, 10), (410, 60), colors["Red"], -1)
    cv2.rectangle(frame, (430, 10), (510, 60), colors["Yellow"], -1)

    #Eraser button
    cv2.rectangle(frame, (530, 10), (630, 60), (255, 255, 255), -1)
    cv2.putText(frame, "ERASER", (540, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2)

    #Brush thickness display
    cv2.putText(frame,
                f"Brush Size: {brush_thickness}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    
    #Process Contours
    if contours:

        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)

        if area > min_area:

            #Find center
            M = cv2.moments(contour)

            if M["m00"] != 0:

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                #Draw tracker circle
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)

                #Button Detection
                if cy <= 60:

                    #Clear canvas
                    if 10 <= cx <= 110:
                        canvas = np.zeros((height, width, 3), dtype=np.uint8)

                    #Blue
                    elif 130 <= cx <= 210:
                        current_color = colors["Blue"]

                    #Green
                    elif 230 <= cx <= 310:
                        current_color = colors["Green"]

                    #Red
                    elif 330 <= cx <= 410:
                        current_color = colors["Red"]

                    #Yellow
                    elif 430 <= cx <= 510:
                        current_color = colors["Yellow"]

                    #Eraser
                    elif 530 <= cx <= 630:
                        current_color = colors["Eraser"]

                    prev_x, prev_y = 0, 0

                else:

                    #Start drawing
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = cx, cy

                    #Draw line
                    cv2.line(canvas,
                             (prev_x, prev_y),
                             (cx, cy),
                             current_color,
                             brush_thickness)

                    prev_x, prev_y = cx, cy

        else:
            prev_x, prev_y = 0, 0


    #Merge Canvas with Frame
    
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    _, mask_inv = cv2.threshold(
        canvas_gray,
        20,
        255,
        cv2.THRESH_BINARY_INV
    )

    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, mask_inv)
    frame = cv2.bitwise_or(frame, canvas)

    
    #Keyboard Controls
    key = cv2.waitKey(1) & 0xFF

    #Quit
    if key == ord('q'):
        break

    #Increase brush size
    elif key == ord('+'):
        brush_thickness += 1

    #Decrease brush size
    elif key == ord('-'):
        brush_thickness = max(1, brush_thickness - 1)

    
    
    cv2.imshow("Air Canvas", frame)
    cv2.imshow("Mask", mask)



cap.release()
cv2.destroyAllWindows()