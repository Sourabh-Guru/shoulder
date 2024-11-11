import cv2
import mediapipe as mp
import numpy as np

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_shoulder_alignment(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if results.pose_landmarks:
            # Get shoulder landmarks
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Convert normalized coordinates to pixel coordinates
            h, w, c = img.shape
            left_y = int(left_shoulder.y * h)
            right_y = int(right_shoulder.y * h)
            
            # Calculate difference in shoulder height
            shoulder_diff = abs(left_y - right_y)
            threshold = 20  # pixels difference threshold
            
            # Draw shoulder points and connection
            left_point = (int(left_shoulder.x * w), left_y)
            right_point = (int(right_shoulder.x * w), right_y)
            cv2.circle(img, left_point, 5, (0, 255, 0), -1)
            cv2.circle(img, right_point, 5, (0, 255, 0), -1)
            cv2.line(img, left_point, right_point, (0, 255, 0), 2)
            
            # Check alignment and display message
            if shoulder_diff > threshold:
                if left_y > right_y:
                    message = "Left Shoulder Not Level"
                    color = (0, 0, 255)  # Red color for warning
                else:
                    message = "Right Shoulder Not Level"
                    color = (0, 0, 255)
            else:
                message = "Shoulders Level"
                color = (0, 255, 0)  # Green color for good alignment
                
            # Display message on image
            cv2.putText(img, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)
            
            return img, message
        
        return img, "No pose detected"

# Main execution
cap = cv2.VideoCapture(0)
detector = PostureDetector()

while True:
    success, img = cap.read()
    if not success:
        break
        
    img, message = detector.detect_shoulder_alignment(img)
    
    cv2.imshow("Shoulder Alignment", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()