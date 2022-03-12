import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from deepface import DeepFace


st.title('Emotion Detection')


try:
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})



class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        
        #Draw a rectangle around the faces
        for(x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 0), thickness=2)
            

            font = cv2.FONT_HERSHEY_SIMPLEX
            result = DeepFace.analyze(img, actions= ['emotion'])

            # Use putText() method for
            # inseting text on video
            label_position = (x, y-25)
            cv2.putText(img,
                    result['dominant_emotion'], 
                    label_position,
                    font, 1,
                    (0,0,255),
                    2,
                    cv2.LINE_4)

            # finalout = result['dominant_emotion']
            # output = str(finalout)     
            # label_position = (x, y)  
            # cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img



def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")
    # activiteis = ["Webcam Face Detection"]
    # choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Sheikh Mohsin Kader  
            Email : smk.mohsin@gmail.com  
            LinkedIn : https://www.linkedin.com/in/smkmohsin/""")

    # else:
    #     pass
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=Faceemotion)

if __name__ == "__main__":
    main()