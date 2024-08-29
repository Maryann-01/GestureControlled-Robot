import cv2
import mediapipe as mp
import pygame
# Initializing MediaPipe Hands
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
# Initialize Pygame
pygame.init()
WIDTH,HEIGHT=1280,480
screen=pygame.display.set_mode((WIDTH,HEIGHT))
# Property of the virtual robot
robot_img = pygame.Surface((50, 50))
robot_img.fill((0, 250, 3))  # Square representing the robot
robot_rect = robot_img.get_rect(center=(WIDTH * 3/4, HEIGHT/2))

# Capture video
cap = cv2.VideoCapture(0)

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()

    # Capture and process webcam frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image of openCV to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #  hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            #The circle located at the index finger tip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # CControlling the robot movement based on hand position
            target_x = (x / frame.shape[1]) * (WIDTH/2) + (WIDTH/2)
            target_y = (y / frame.shape[0]) * HEIGHT

            # Move robot towards the target position
            robot_rect.x += (target_x - robot_rect.centerx) * 0.1
            robot_rect.y += (target_y - robot_rect.centery) * 0.1

    # Convert frame to Pygame surface
    webcam_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))

    # Clear the screen
    screen.fill((255, 255, 255))
    # Draw webcam feed on the left side
    screen.blit(webcam_surface, (0, 0))
    # Draw a line separating webcam feed and robot area
    pygame.draw.line(screen, (0, 0, 0), (WIDTH/2, 0), (WIDTH/2, HEIGHT), 2)
    # Draw robot
    screen.blit(robot_img, robot_rect)
    # Update the display
    pygame.display.flip()
    # Cap the frame rate
    clock.tick(60)

cap.release()
pygame.quit()
