import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
import onnxruntime as ort
import yaml
from yaml.loader import SafeLoader
import joblib
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

st.markdown('<p class="big-font">✈️Arcade Flyers</p>', unsafe_allow_html=True) 
st.markdown('<p class="small-font">Fly Safe, Land Secure..!!</p>', unsafe_allow_html=True)

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Load YAML labels
def load_labels(yaml_path):
    with open(yaml_path, mode='r') as f:
        data_yaml = yaml.load(f, Loader=SafeLoader)
    return data_yaml['names']

# Load ONNX model
def load_model(model_path):
    return ort.InferenceSession(model_path)

# Preprocess image for YOLO
def preprocess_image(image):
    image = image.copy()
    max_rc = max(image.shape[:2])
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:image.shape[0], 0:image.shape[1]] = image
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True, crop=False)
    return input_image, blob

# Postprocess detections from YOLO
def postprocess_detections(preds, input_image_shape, labels):
    detections = preds[0]
    boxes, confidences, classes = [], [], []
    image_w, image_h = input_image_shape[:2]
    x_factor = image_w / 640
    y_factor = image_h / 640

    for row in detections:
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5:].max()
            class_id = row[5:].argmax()
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                box = np.array([left, top, int(w * x_factor), int(h * y_factor)])
                boxes.append(box)
                confidences.append(confidence)
                classes.append(class_id)

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()
    return [(boxes[i], confidences[i], labels[classes[i]]) for i in indices]

# Detect dents and cracks
def detect_dents_and_cracks(image, model, labels):
    input_image, blob = preprocess_image(image)
    preds = model.run([model.get_outputs()[0].name], {model.get_inputs()[0].name: blob})[0]
    return postprocess_detections(preds, input_image.shape, labels)

# Custom CSS
def custom_css():
    st.markdown("""
        <style>
        .appview-container {
            background-image: linear-gradient(to right top, #b3dce3, #85c6dc, #57afd9, #2c96d5, #167bce);
        }
        .big-font {
            font-size:50px !important;
            font-weight: bold;
            color: white;
        }
        .small-font {
            font-family: monospace;
            font-weight: 200;
            font-style: italic;
            font-size:20px !important;
        }
        .normal-font {
            font-family: fangsong;
            font-weight: 800;
            font-size: 30px !important;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

# Vector embedding and question answering
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./data_report")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector store
        st.success("Vector Store Created")

# Main app
def main():
    # Load labels and models
    labels = load_labels('data.yaml')
    yolo_model = load_model('Model/weights/best.onnx')
    Fault = joblib.load('Model/Wire_Fault.joblib')
    
    custom_css()
    
    selected = option_menu(menu_title=None,
            options=['Damage Detection', 'Faulty Wire Detection', 'Fault Bot'],
            icons=['activity', 'activity', 'question-circle'],
            default_index=0,
            orientation='horizontal',
            menu_icon="cast")

    if selected == "Damage Detection":
        st.markdown('<p class="normal-font">Structural Integrity Assessment</p>', unsafe_allow_html=True)
        st.info('Upload an image to check for any dents or cracks.', icon="ℹ️")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
                st.image(image, caption="Original Image", use_column_width=True)

                with st.spinner('Processing...'):
                    damage_locations = detect_dents_and_cracks(image, yolo_model, labels)

                for (box, confidence, class_name) in damage_locations:
                    left, top, width, height = box
                    st.write(f"{class_name} detected with {confidence:.2f}% confidence at ({left}, {top})")
                    cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

                st.image(image, caption="Marked Image", use_column_width=True)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    if selected == "Faulty Wire Detection":
        st.markdown('<p class="normal-font">Electrical Fault Identification</p>', unsafe_allow_html=True)
        st.info('Enter data to check for any faults.', icon="ℹ️")
        v = st.text_input("Enter Voltage:")
        i = st.text_input("Enter Current:")
        r = st.text_input("Enter Resistance")

        def predict_wire_status(v, i, r):
            try:
                prediction = Fault.predict([[float(v), float(i), float(r)]])
                return prediction[0]
            except ValueError:
                st.error("Please enter valid numerical values.")
                return None

        if st.button('Predict'):
            result = predict_wire_status(v, i, r)
            if result is not None:
                if result == 1:
                    st.write('__Faulty Wire Detected__')
                else:
                    st.write('__No Fault__')

    if selected == "Fault Bot":
        st.title("Gen AI- RAG for Airplane Analytics")
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0.2, top_p=0.7, max_tokens=1024)
        
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )

        prompt1 = st.text_input("Enter Your Question related to airplane accidents")

        if st.button("Documents Embedding"):
            vector_embedding()

        # st.write("Vector Store DB Is Ready")

        if prompt1:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", time.process_time() - start)
            st.write(response['answer'])

            # Document Similarity Search
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

if __name__ == "__main__":
    main()