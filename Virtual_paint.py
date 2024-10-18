import mediapipe as mp
import cv2
import numpy as np
import time

# constants
ml = 150  # Margin left for tools
max_x, max_y = 250+ml, 50  # Define tool selection area dimensions
curr_tool = "select tool"  # Current tool to be selected by hand gestures
time_init = True  # To track the initial time for tool selection
rad = 40  # Radius of the circle for selecting a tool
var_inits = False  # Variable to track initialization of shapes like line, rectangle
thick = 4  # Thickness of drawing strokes or shapes
prevx, prevy = 0, 0  # Previous coordinates for drawing

# get tools function
def getTool(x):
    # Select tool based on x-coordinate in tool area
    if x < 50 + ml:
        return "line"
    elif x < 100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return "draw"
    elif x < 200 + ml:
        return "circle"
    else:
        return "erase"

# Check if index finger is raised for drawing/shaping
def index_raised(yi, y9):
    if (y9 - yi) > 40:
        return True
    return False

hands = mp.solutions.hands  # Use MediaPipe Hands for hand tracking
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils  # Draw hand landmarks

# Load drawing tools image
tools = cv2.imread("tools.png")  # Load image containing tool options
tools = tools.astype('uint8')  # Convert tools image to 8-bit unsigned integer

# Create a white mask to draw on
mask = np.ones((480, 640)) * 255  # White canvas for drawing
mask = mask.astype('uint8')  # Convert mask to 8-bit unsigned integer

cap = cv2.VideoCapture(0)  # Start video capture from webcam

# Create a named window and set it to full screen
cv2.namedWindow("paint app", cv2.WINDOW_NORMAL)  # Create resizable window
cv2.setWindowProperty("paint app", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set window to full screen

while True:
    _, frm = cap.read()  # Capture video frame-by-frame
    frm = cv2.flip(frm, 1)  # Flip frame horizontally to mirror camera

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for MediaPipe processing

    op = hand_landmark.process(rgb)  # Process hand landmarks using MediaPipe

    if op.multi_hand_landmarks:
        # Loop through detected hand landmarks
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)  # Draw hand landmarks on the frame
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)  # Get coordinates of index fingertip

            # Check if the fingertip is in the tool area
            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()  # Start timing for tool selection
                    time_init = False
                ptime = time.time()

                cv2.circle(frm, (x, y), rad, (0, 255, 255), 2)  # Draw circle around fingertip in tool area
                rad -= 1  # Shrink circle as time progresses

                if (ptime - ctime) > 0.8:  # Select tool if circle shrinks for 0.8 seconds
                    curr_tool = getTool(x)
                    print("your current tool set to : ", curr_tool)
                    time_init = True
                    rad = 40  # Reset radius after tool selection

            else:
                time_init = True  # Reset timing when hand leaves tool area
                rad = 40  # Reset circle radius when outside tool area

            if curr_tool == "draw":
                # Drawing mode, track index and thumb for drawing
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):  # Draw when index finger is raised
                    cv2.line(mask, (prevx, prevy), (x, y), 0, thick)  # Draw line on mask
                    prevx, prevy = x, y
                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "line":
                # Line mode, track starting and ending points for lines
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:  # Initialize line's start point
                        xii, yii = x, y
                        var_inits = True
                    cv2.line(frm, (xii, yii), (x, y), (50, 152, 255), thick)  # Draw line preview on frame
                else:
                    if var_inits:  # Finalize and draw line on mask
                        cv2.line(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "rectangle":
                # Rectangle mode, track starting and ending points for rectangles
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:  # Initialize rectangle's starting point
                        xii, yii = x, y
                        var_inits = True
                    cv2.rectangle(frm, (xii, yii), (x, y), (0, 255, 255), thick)  # Draw rectangle preview on frame
                else:
                    if var_inits:  # Finalize and draw rectangle on mask
                        cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
                        var_inits = False

            elif curr_tool == "circle":
                # Circle mode, track radius for drawing circles
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    if not var_inits:  # Initialize circle's center point
                        xii, yii = x, y
                        var_inits = True
                    cv2.circle(frm, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (255, 255, 0), thick)  # Draw circle preview on frame
                else:
                    if var_inits:  # Finalize and draw circle on mask
                        cv2.circle(mask, (xii, yii), int(((xii - x) ** 2 + (yii - y) ** 2) ** 0.5), (0, 255, 0), thick)
                        var_inits = False

            elif curr_tool == "erase":
                # Erase mode, erase portions of the drawing
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):  # Erase when index finger is raised
                    cv2.circle(frm, (x, y), 30, (0, 0, 0), -1)  # Erase from frame
                    cv2.circle(mask, (x, y), 30, 255, -1)  # Erase from mask

    # Combine drawing with frame
    op = cv2.bitwise_and(frm, frm, mask=mask)  # Apply the mask to the frame
    frm[:, :, 1] = op[:, :, 1]
    frm[:, :, 2] = op[:, :, 2]

    # Add tool image to frame
    frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)

    cv2.putText(frm, curr_tool, (270+ml, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Display current tool
    cv2.imshow("paint app", frm)  # Show the frame in full screen

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        cv2.destroyAllWindows()
        cap.release()
        break
