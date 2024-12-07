import cv2
import streamlit as st
import numpy as np
import time
import os

# Set page layout to wide
st.set_page_config(layout="wide")

# Initialize session state variables
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False
if "photo_taken" not in st.session_state:
    st.session_state["photo_taken"] = False  # To track if photo has already been taken

# Streamlit UI
st.title("Real-Time Face Detection")

# Descrição do projeto
st.markdown("""
    Este projeto utiliza técnicas de visão computacional para detectar rostos em tempo real a partir de uma webcam. O sistema é baseado em OpenCV, uma poderosa biblioteca de visão computacional, e pode identificar tanto rostos frontais quanto de perfil.
    
    O aplicativo foi desenvolvido com o objetivo de demonstrar como a detecção facial pode ser realizada utilizando classificadores Haar para detectar faces a partir de imagens de vídeo em tempo real.
    
    #### Funcionalidades:
    - Detecção de rostos frontais e de perfil.
    - Interface interativa com Streamlit.
    - Ativação e desativação da câmera diretamente pela interface.
    - Captura automática de foto quando um rosto é detectado (apenas uma vez).
""")

# Adicionando uma linha horizontal ocupando toda a página
st.markdown(
    "<hr style='border: 1px solid #000; width: 100%; margin: 10px 0;'>", 
    unsafe_allow_html=True
)

# CSS para diminuir a largura da barra de progresso
st.markdown(
    """
    <style>
    .stProgress > div {
        width: 50% !important;  /* Ajuste o valor conforme necessário */
        margin: 1 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use a single column for buttons (this can reduce the space)
col1, col2 = st.columns([1, 1])  # Two columns for camera and image

# Button to open the camera
with col1:
    if st.button("Open Camera"):
        # Show loading bar and text immediately
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.text("Opening camera... Please wait.")
        
        # Simulate camera opening process with progress bar
        for i in range(1, 101):
            time.sleep(0.03)  # Simulate loading time
            progress_bar.progress(i)  # Update progress bar

        # Once progress is complete, activate the camera
        st.session_state["camera_active"] = True
        progress_text.text("Camera is now active.")

    if st.button("Close Camera"):
        st.session_state["camera_active"] = False

# Inicializa a variável image_path com um valor vazio
image_path = ""

# If the camera is active, start the detection
if st.session_state["camera_active"]:
    # Create the Haar cascade classifier for face detection (front and profile)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Placeholder for displaying the video frames
    frame_placeholder = col1.empty()
    image_placeholder = col2.empty()  # Placeholder for displaying the captured photo

    # Create a directory to save the captured images
    if not os.path.exists('captured_faces'):
        os.makedirs('captured_faces')

    # Run the camera loop
    while st.session_state["camera_active"]:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            st.write("Failed to capture video")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame (front faces)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Detect profile faces (faces from the side)
        profile_faces = profileCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Expansão da área de detecção para incluir o pescoço
        margin = 35  # Ajuste esse valor conforme necessário para expandir a detecção
        neck_margin = 45  # Ajuste esse valor para capturar o pescoço

        # Para rostos frontais
        for (x, y, w, h) in faces:
            # Expandindo a área de detecção para incluir o pescoço
            y = max(y - margin, 0)  # Garantir que não ultrapasse a borda superior
            w = w + 2 * margin  # Aumentar a largura
            h = h + margin + neck_margin  # Aumentar a altura, incluindo o pescoço

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture a nova área (que inclui o pescoço)
            if not st.session_state["photo_taken"]:
                face_image = frame[y:y+h, x:x+w]
                image_path = os.path.join('captured_faces', f'face_with_neck_{int(time.time())}.jpg')
                cv2.imwrite(image_path, face_image)  # Salve a imagem
                st.write(f"Face detected with neck! Photo saved as: {image_path}")
                st.session_state["photo_taken"] = True  # Marque que a foto foi tirada

        # Para rostos de perfil
        for (x, y, w, h) in profile_faces:
            # Expandindo a área de detecção para incluir o pescoço
            y = max(y - margin, 0)
            w = w + 2 * margin
            h = h + margin + neck_margin  # Aumentar a altura para incluir o pescoço

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not st.session_state["photo_taken"]:
                face_image = frame[y:y+h, x:x+w]
                image_path = os.path.join('captured_faces', f'profile_face_with_neck_{int(time.time())}.jpg')
                cv2.imwrite(image_path, face_image)
                st.write(f"Profile face with neck detected! Photo saved as: {image_path}")
                st.session_state["photo_taken"] = True

        # Convert the frame to RGB (because Streamlit expects RGB images)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame in the Streamlit app (left side column)
        frame_placeholder.image(frame, channels="RGB")

        # Display the captured photo in the right column
        if st.session_state["photo_taken"]:
            image_placeholder.image(image_path)  # Show the captured face photo

        # Stop the loop if the user clicks "Close Camera"
        if not st.session_state["camera_active"]:
            break

    # Release the capture when done
    cap.release()
    st.write("Camera stopped.")
