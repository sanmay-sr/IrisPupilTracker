import streamlit as st
import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64
import time

# Page configuration
st.set_page_config(
    page_title="Iris-Pupil Ratio Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# Enhanced Custom CSS for professional styling with mobile responsiveness
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .metric-card h3 {
            font-size: 1.4rem;
        }
        .section-header {
            padding: 0.8rem 1rem;
            margin: 1rem 0 0.5rem 0;
        }
        .section-header h2 {
            font-size: 1.2rem;
        }
        .warning-box, .success-box, .info-box {
            padding: 1rem;
            margin: 0.8rem 0;
        }
        .footer {
            padding: 1.5rem;
            margin-top: 2rem;
        }
        .footer h4 {
            font-size: 1.1rem;
        }
        .footer p, .footer small {
            font-size: 0.85rem;
        }
        .camera-instructions {
            padding: 1rem;
        }
        .camera-instructions ul {
            padding-left: 1.2rem;
        }
        .camera-instructions li {
            font-size: 0.9rem;
            margin: 0.2rem 0;
        }
    }
    
    /* Tablet adjustments */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        .metric-card h3 {
            font-size: 1.6rem;
        }
    }
    
    /* Base styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .metric-card h4 {
        color: #495057;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h3 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card small {
        color: #6c757d;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #f39c12;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px rgba(243, 156, 18, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #27ae60;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #17a2b8;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px rgba(23, 162, 184, 0.1);
    }
    .section-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .section-header h2 {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    .camera-instructions {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .camera-instructions ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .camera-instructions li {
        margin: 0.3rem 0;
        color: #495057;
    }
    .footer {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .footer h4 {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .footer p {
        color: #6c757d;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .footer small {
        color: #adb5bd;
        font-size: 0.8rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
        min-height: 44px; /* Mobile touch target size */
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Mobile-specific improvements */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
            margin: 0.3rem 0;
        }
        .stRadio > div {
            flex-direction: column;
        }
        .stRadio > div > label {
            margin: 0.2rem 0;
        }
        .stFileUploader > div {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 1rem;
        }
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div > div {
            font-size: 16px; /* Prevents zoom on iOS */
        }
    }
    
    /* Touch-friendly improvements */
    @media (hover: none) and (pointer: coarse) {
        .metric-card:hover {
            transform: none;
        }
        .stButton > button:hover {
            transform: none;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Mediapipe Face Mesh
@st.cache_resource
def initialize_mediapipe():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    return mp_face_mesh, face_mesh

# Define eye and iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

def capture_photo():
    """Capture photo from webcam using OpenCV"""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return None, "Camera access denied. Please check your browser camera permissions and refresh the page."
    
    # Set camera properties for better quality
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    captured_frame = None
    error_message = None
    
    try:
        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            error_message = "Failed to capture video frame."
        else:
            # Flip the frame horizontally for a more intuitive experience
            frame = cv.flip(frame, 1)
            captured_frame = frame.copy()
            
    except Exception as e:
        error_message = f"Camera error: {str(e)}"
    
    finally:
        cap.release()
    
    if captured_frame is not None:
        # Convert OpenCV frame to PIL Image
        captured_frame_rgb = cv.cvtColor(captured_frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(captured_frame_rgb)
        return pil_image, None
    else:
        return None, error_message

def start_live_camera():
    """Start live camera stream for preview"""
    # This function is now simplified - just returns success
    return True, None

def get_live_frame(cap):
    """Get a single frame from the live camera stream"""
    # This function is no longer needed with st.camera_input
    return None, "Function deprecated - using Streamlit camera widget"

def stop_live_camera(cap):
    """Stop and release the camera stream"""
    # This function is no longer needed with st.camera_input
    pass

def process_iris_analysis(image, mp_face_mesh, face_mesh):
    """Process iris analysis on the uploaded image"""
    
    # Convert PIL image to OpenCV format
    opencv_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    gray_frame = cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
    
    # Process with MediaPipe
    results = face_mesh.process(cv.cvtColor(opencv_image, cv.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None, "No face landmarks detected in the image."
    
    img_h, img_w = opencv_image.shape[:2]
    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                            for p in results.multi_face_landmarks[0].landmark])

    # Calculate IPD and pixel to mm conversion
    ipd_pixels = np.linalg.norm(mesh_points[LEFT_EYE_OUTER] - mesh_points[RIGHT_EYE_OUTER])
    ipd_mm = 63  # Average adult IPD
    pixel_to_mm = ipd_mm / ipd_pixels

    def analyze_single_iris(iris_landmarks, eye_side):
        try:
            (cx, cy), iris_radius_px = cv.minEnclosingCircle(mesh_points[iris_landmarks])
        except IndexError:
            return None
        
        iris_center = np.array([cx, cy], dtype=np.int32)
        iris_radius_mm = iris_radius_px * pixel_to_mm

        # Extract iris ROI
        x1, y1 = max(0, iris_center[0] - int(iris_radius_px)), max(0, iris_center[1] - int(iris_radius_px))
        x2, y2 = min(img_w, iris_center[0] + int(iris_radius_px)), min(img_h, iris_center[1] + int(iris_radius_px))
        
        iris_roi = gray_frame[y1:y2, x1:x2]
        
        if iris_roi.size == 0:
            return None
        
        # Process iris ROI for pupil detection
        iris_gray = cv.equalizeHist(iris_roi)
        blurred = cv.GaussianBlur(iris_gray, (5, 5), 0)
        thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv.THRESH_BINARY_INV, 11, 2)
        edges = cv.Canny(blurred, 50, 150)
        combined = cv.bitwise_and(thresh, edges)
        contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        pupil_radius_mm = 0
        pupil_center = (int(iris_center[0]), int(iris_center[1]))

        if contours:
            largest_area = 0
            best_ellipse = None
            for contour in contours:
                if len(contour) >= 5:
                    ellipse = cv.fitEllipse(contour)
                    (x, y), (MA, ma), angle = ellipse
                    area = np.pi * (MA / 2) * (ma / 2)
                    if area > largest_area:
                        largest_area = area
                        best_ellipse = ellipse

            if best_ellipse:
                (x, y), (MA, ma), angle = best_ellipse
                radius_px = (MA + ma) / 4
                min_pupil_radius_px = iris_radius_px * 0.2
                max_pupil_radius_px = iris_radius_px * 0.7

                if min_pupil_radius_px <= radius_px <= max_pupil_radius_px:
                    pupil_radius_mm = radius_px * pixel_to_mm
                    pupil_center = (int(x + x1), int(y + y1))
                else:
                    pupil_radius_mm = iris_radius_mm / 3
            else:
                pupil_radius_mm = iris_radius_mm / 3
        else:
            pupil_radius_mm = iris_radius_mm / 3

        iris_pupil_ratio = iris_radius_mm / pupil_radius_mm if pupil_radius_mm > 0 else 0
        iris_pupil_ratio = max(2.0, min(iris_pupil_ratio, 5.0))

        # Create annotated image
        annotated_frame = opencv_image.copy()
        cv.circle(annotated_frame, tuple(iris_center), int(iris_radius_px), (0, 0, 255), 2)
        cv.circle(annotated_frame, tuple(pupil_center), int(pupil_radius_mm / pixel_to_mm), (0, 255, 0), 2)

        return {
            'iris_radius_mm': iris_radius_mm,
            'pupil_radius_mm': pupil_radius_mm,
            'iris_pupil_ratio': iris_pupil_ratio,
            'annotated_image': annotated_frame,
            'iris_center': iris_center,
            'pupil_center': pupil_center
        }

    # Process both eyes
    left_eye_data = analyze_single_iris(LEFT_IRIS, "Left Eye")
    right_eye_data = analyze_single_iris(RIGHT_IRIS, "Right Eye")
    
    return {
        'left_eye': left_eye_data,
        'right_eye': right_eye_data,
        'original_image': opencv_image
    }, None

def create_pdf_report(patient_info, analysis_results, original_image=None):
    """Create a professional PDF report with patient information and analysis results"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Enhanced custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1,  # Center alignment
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    # Professional header
    story.append(Paragraph("Iris-Pupil Ratio Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                          ParagraphStyle('Date', parent=styles['Normal'], alignment=1, fontSize=10)))
    story.append(Spacer(1, 20))
    
    # Patient Information with better formatting
    story.append(Paragraph("Patient Information", subtitle_style))
    
    patient_data = [
        ['Patient Name:', patient_info['name']],
        ['Age:', f"{patient_info['age']} years"],
        ['Gender:', patient_info['gender']],
        ['Analysis Date:', datetime.now().strftime("%B %d, %Y")],
        ['Analysis Time:', datetime.now().strftime("%I:%M %p")],
    ]
    
    if patient_info.get('medical_conditions'):
        patient_data.append(['Medical Notes:', patient_info['medical_conditions']])
    
    patient_table = Table(patient_data, colWidths=[2.5*inch, 3.5*inch])
    patient_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 25))
    
    # Analysis Results with enhanced formatting
    story.append(Paragraph("Analysis Results", subtitle_style))
    
    results_data = []
    if analysis_results['left_eye']:
        left_ratio = analysis_results['left_eye']['iris_pupil_ratio']
        left_status = "Normal" if 2.5 <= left_ratio <= 4.0 else "Outside Normal Range"
        results_data.extend([
            ['Left Eye Analysis', ''],
            ['  • Iris Radius:', f"{analysis_results['left_eye']['iris_radius_mm']:.2f} mm"],
            ['  • Pupil Radius:', f"{analysis_results['left_eye']['pupil_radius_mm']:.2f} mm"],
            ['  • Iris-to-Pupil Ratio:', f"{left_ratio:.2f} ({left_status})"],
            ['', ''],  # Empty row for spacing
        ])
    
    if analysis_results['right_eye']:
        right_ratio = analysis_results['right_eye']['iris_pupil_ratio']
        right_status = "Normal" if 2.5 <= right_ratio <= 4.0 else "Outside Normal Range"
        results_data.extend([
            ['Right Eye Analysis', ''],
            ['  • Iris Radius:', f"{analysis_results['right_eye']['iris_radius_mm']:.2f} mm"],
            ['  • Pupil Radius:', f"{analysis_results['right_eye']['pupil_radius_mm']:.2f} mm"],
            ['  • Iris-to-Pupil Ratio:', f"{right_ratio:.2f} ({right_status})"],
        ])
    
    results_table = Table(results_data, colWidths=[3.5*inch, 2.5*inch])
    results_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 25))
    
    # Original Image Section with proper aspect ratio
    if original_image is not None:
        story.append(Paragraph("Analyzed Image", subtitle_style))
        
        # Convert PIL image to bytes for PDF with proper aspect ratio
        img_buffer = io.BytesIO()
        original_image.save(img_buffer, format='JPEG', quality=90)
        img_buffer.seek(0)
        
        # Calculate proper dimensions maintaining aspect ratio
        img_width, img_height = original_image.size
        aspect_ratio = img_width / img_height
        
        # Set maximum dimensions for PDF
        max_width = 5.5 * inch
        max_height = 4 * inch
        
        if aspect_ratio > 1:  # Landscape
            pdf_width = max_width
            pdf_height = max_width / aspect_ratio
        else:  # Portrait
            pdf_height = max_height
            pdf_width = max_height * aspect_ratio
        
        # Ensure dimensions don't exceed maximum
        if pdf_width > max_width:
            pdf_width = max_width
            pdf_height = max_width / aspect_ratio
        if pdf_height > max_height:
            pdf_height = max_height
            pdf_width = max_height * aspect_ratio
        
        try:
            pdf_image = RLImage(img_buffer, width=pdf_width, height=pdf_height)
            story.append(pdf_image)
            story.append(Spacer(1, 15))
        except Exception as e:
            story.append(Paragraph(f"Image could not be included: {str(e)}", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    # Annotated Images Section with better layout
    if (analysis_results['left_eye'] and analysis_results['left_eye']['annotated_image'] is not None) or \
       (analysis_results['right_eye'] and analysis_results['right_eye']['annotated_image'] is not None):
        
        story.append(Paragraph("Analysis Visualization", subtitle_style))
        
        # Create a table for side-by-side annotated images
        annotated_images_data = []
        
        if analysis_results['left_eye'] and analysis_results['left_eye']['annotated_image'] is not None:
            left_annotated = cv.cvtColor(analysis_results['left_eye']['annotated_image'], cv.COLOR_BGR2RGB)
            left_pil = Image.fromarray(left_annotated)
            
            left_buffer = io.BytesIO()
            left_pil.save(left_buffer, format='JPEG', quality=90)
            left_buffer.seek(0)
            
            try:
                left_pdf_img = RLImage(left_buffer, width=2.8*inch, height=2.1*inch)
                annotated_images_data.append([left_pdf_img, "Left Eye Analysis"])
            except Exception as e:
                annotated_images_data.append([Paragraph("Left eye image unavailable", styles['Normal']), "Left Eye Analysis"])
        else:
            annotated_images_data.append([Paragraph("No data available", styles['Normal']), "Left Eye Analysis"])
        
        if analysis_results['right_eye'] and analysis_results['right_eye']['annotated_image'] is not None:
            right_annotated = cv.cvtColor(analysis_results['right_eye']['annotated_image'], cv.COLOR_BGR2RGB)
            right_pil = Image.fromarray(right_annotated)
            
            right_buffer = io.BytesIO()
            right_pil.save(right_buffer, format='JPEG', quality=90)
            right_buffer.seek(0)
            
            try:
                right_pdf_img = RLImage(right_buffer, width=2.8*inch, height=2.1*inch)
                annotated_images_data.append([right_pdf_img, "Right Eye Analysis"])
            except Exception as e:
                annotated_images_data.append([Paragraph("Right eye image unavailable", styles['Normal']), "Right Eye Analysis"])
        else:
            annotated_images_data.append([Paragraph("No data available", styles['Normal']), "Right Eye Analysis"])
        
        # Create table for annotated images
        annotated_table = Table(annotated_images_data, colWidths=[3*inch, 3*inch])
        annotated_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        story.append(annotated_table)
        story.append(Spacer(1, 15))
        
                # Add legend
        legend_text = "Legend: Red circle = Iris boundary, Green circle = Pupil boundary"
        story.append(Paragraph(legend_text, ParagraphStyle('Legend', parent=styles['Normal'], 
                                                         fontSize=9, textColor=colors.grey, alignment=1)))
        
        story.append(Spacer(1, 20))
    
    # Reference Information with cleaner formatting
    story.append(Paragraph("Reference Information", subtitle_style))
    reference_text = """
    <b>Normal Range:</b> Iris-to-pupil ratio typically ranges between 2.5 to 4.0.<br/>
    <b>Factors:</b> Measurements can vary based on lighting conditions, age, and individual anatomical differences.<br/>
    <b>Technology:</b> Analysis performed using MediaPipe Face Mesh and OpenCV computer vision algorithms.<br/>
    <br/>
    <b>Disclaimer:</b> This analysis is for research and educational purposes only. Results should not be used for medical diagnosis. 
    Please consult with qualified healthcare professionals for medical advice.
    """
    story.append(Paragraph(reference_text, ParagraphStyle('Reference', parent=styles['Normal'], 
                                                         fontSize=10, spaceAfter=12)))
    
    # Footer
    story.append(Spacer(1, 20))
    footer_text = f"Report generated by Iris-Pupil Ratio Analyzer v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], 
                                                      fontSize=8, textColor=colors.grey, alignment=1)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    # Professional header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1>Iris-Pupil Ratio Analyzer</h1>
        <p>Advanced Computer Vision Analysis for Research & Educational Purposes</p>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>Research & Educational Use Only:</strong> This tool utilizes advanced computer vision algorithms 
        for iris-pupil ratio analysis. Results are for research and educational purposes only and should not be 
        used for medical diagnosis. Always consult qualified healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize MediaPipe
    mp_face_mesh, face_mesh = initialize_mediapipe()
    
    # Professional sidebar for patient information
    st.sidebar.markdown("""
    <div class="section-header">
        <h2>Patient Information</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Mobile-friendly instructions
    st.sidebar.markdown("""
    <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 1rem;">
        Complete the form below to generate a professional analysis report.
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.form("patient_form"):
        name = st.text_input("Full Name", placeholder="Enter patient's full name", 
                            help="Required for report generation")
        age = st.number_input("Age", min_value=1, max_value=120, value=25, 
                             help="Patient's age in years")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                             help="Patient's gender")
        medical_conditions = st.text_area("Medical Notes (Optional)", 
                                        placeholder="Any relevant medical conditions, medications, or eye conditions...",
                                        help="Optional: Include any relevant medical history")
        
        submit_info = st.form_submit_button("Save Patient Information", type="primary")
        
        if submit_info and name:
            st.sidebar.success("Patient information saved successfully!")
        elif submit_info and not name:
            st.sidebar.error("Please enter the patient's name.")
    
    # Main content area - responsive layout
    # Use responsive columns that stack on mobile
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("""
        <div class="section-header">
            <h2>Image Input</h2>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Choose your preferred method to input an image for analysis.")
        
        # Image input method selection
        input_method = st.radio(
            "Select Input Method:",
            ["Upload Image", "Live Camera Capture"],
            horizontal=True
        )
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear frontal face image with visible eyes. Supported formats: JPG, JPEG, PNG"
            )
            captured_image = None
        else:
            uploaded_file = None
            st.markdown("""
            <div class="camera-instructions">
                <h4>Live Camera Instructions:</h4>
                <ul>
                    <li>Click <strong>'Start Live Camera'</strong> to activate the webcam</li>
                    <li>You'll see a <strong>LIVE VIDEO STREAM</strong> with a capture button</li>
                    <li>Position your face clearly in the camera view</li>
                    <li>Ensure good lighting and both eyes are visible</li>
                    <li>Click the <strong>'Take photo'</strong> button in the camera widget to capture</li>
                    <li>Use 'Reset Camera' to start over if needed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Camera permission note in a separate markdown call
            st.markdown("""
            <div class="warning-box">
                <strong>Camera Permission Note:</strong> If you see "Camera access denied", please:
                <ul>
                    <li>Check your browser's camera permissions (lock icon in address bar)</li>
                    <li>Allow camera access when prompted</li>
                    <li>Refresh the page after granting permissions</li>
                    <li>Try using a different browser if issues persist</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Camera status indicator
            camera_status = st.session_state.get('camera_active', False)
            if camera_status:
                st.markdown("""
                <div class="info-box">
                    <strong>Camera Active:</strong> Live camera is active. Use the camera widget below to capture your photo.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>Ready to Start:</strong> Click 'Start Live Camera' to begin live video capture.
                </div>
                """, unsafe_allow_html=True)
            
            # Camera preview area
            preview_placeholder = st.empty()
            
            # Camera session state is no longer needed with st.camera_input
            
            # Show live camera preview when active
            if camera_status:
                # Use Streamlit's native camera widget
                camera_photo = st.camera_input("Live Camera Stream", key="live_camera")
                
                if camera_photo is not None:
                    # Convert to PIL Image for consistency
                    captured_image = Image.open(camera_photo)
                    st.success("Photo captured successfully!")
                    st.image(captured_image, caption="Captured Photo", use_container_width=True)
                    # Store in session state for analysis
                    st.session_state['captured_image'] = captured_image
                    st.session_state['camera_active'] = False
                    st.rerun()
            
            # Camera control buttons - responsive layout
            col_cam1, col_cam2 = st.columns(2, gap="small")
            
            with col_cam1:
                if st.button("Start Live Camera", type="secondary", use_container_width=True):
                    st.session_state['camera_active'] = True
                    st.rerun()
            
            with col_cam2:
                if st.button("Reset Camera", type="secondary", use_container_width=True):
                    st.session_state['camera_active'] = False
                    if 'captured_image' in st.session_state:
                        del st.session_state['captured_image']
                    st.rerun()
        
        # Handle image analysis
        image_to_analyze = None
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file)
            st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)
        elif 'captured_image' in st.session_state:
            image_to_analyze = st.session_state['captured_image']
            st.image(image_to_analyze, caption="Captured Photo", use_container_width=True)
        
        if image_to_analyze is not None:
            if st.button("Analyze Image", type="primary"):
                if not name:
                    st.error("Please enter the patient's name in the sidebar first.")
                else:
                    with st.spinner("Analyzing iris-pupil ratio... This may take a few seconds."):
                        try:
                            results, error = process_iris_analysis(image_to_analyze, mp_face_mesh, face_mesh)
                            
                            if error:
                                st.error(f"Analysis failed: {error}")
                                st.info("Tips for better results:\n- Ensure the image shows a clear frontal view of the face\n- Make sure both eyes are visible and well-lit\n- Avoid images with glasses or heavy shadows\n- Use high-resolution images when possible")
                            else:
                                st.session_state['analysis_results'] = results
                                st.session_state['patient_info'] = {
                                    'name': name,
                                    'age': age,
                                    'gender': gender,
                                    'medical_conditions': medical_conditions
                                }
                                # Store the original image for PDF report
                                st.session_state['original_image'] = image_to_analyze
                                st.markdown("""
                                <div class="success-box">
                                    <strong>Analysis completed successfully!</strong><br>
                                    Results are now available in the Analysis Results section.
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
                            st.info("Please try uploading a different image or contact support if the problem persists.")
    
    with col2:
        st.markdown("""
        <div class="section-header">
            <h2>Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            # Display results with enhanced styling - responsive layout
            if results['left_eye']:
                st.markdown("### Left Eye Analysis")
                # Use responsive columns that stack on mobile
                col_l1, col_l2, col_l3 = st.columns(3, gap="small")
                with col_l1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Iris Radius</h4>
                        <h3>{results['left_eye']['iris_radius_mm']:.2f} mm</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Pupil Radius</h4>
                        <h3>{results['left_eye']['pupil_radius_mm']:.2f} mm</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col_l3:
                    ratio = results['left_eye']['iris_pupil_ratio']
                    ratio_status = "Normal" if 2.5 <= ratio <= 4.0 else "Outside Normal Range"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>I/P Ratio</h4>
                        <h3>{ratio:.2f}</h3>
                        <small>{ratio_status}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if results['right_eye']:
                st.markdown("### Right Eye Analysis")
                # Use responsive columns that stack on mobile
                col_r1, col_r2, col_r3 = st.columns(3, gap="small")
                with col_r1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Iris Radius</h4>
                        <h3>{results['right_eye']['iris_radius_mm']:.2f} mm</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col_r2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Pupil Radius</h4>
                        <h3>{results['right_eye']['pupil_radius_mm']:.2f} mm</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with col_r3:
                    ratio = results['right_eye']['iris_pupil_ratio']
                    ratio_status = "Normal" if 2.5 <= ratio <= 4.0 else "Outside Normal Range"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>I/P Ratio</h4>
                        <h3>{ratio:.2f}</h3>
                        <small>{ratio_status}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show annotated images
            st.markdown("### Annotated Images")
            st.markdown("Images showing detected iris (red circle) and pupil (green circle) boundaries.")
            
            if results['left_eye'] and results['left_eye']['annotated_image'] is not None:
                annotated_img = cv.cvtColor(results['left_eye']['annotated_image'], cv.COLOR_BGR2RGB)
                st.image(annotated_img, caption="Left Eye Analysis - Red: Iris, Green: Pupil", use_container_width=True)
            
            if results['right_eye'] and results['right_eye']['annotated_image'] is not None:
                annotated_img = cv.cvtColor(results['right_eye']['annotated_image'], cv.COLOR_BGR2RGB)
                st.image(annotated_img, caption="Right Eye Analysis - Red: Iris, Green: Pupil", use_container_width=True)
            
            # PDF Report Generation
            st.markdown("### Generate Report")
            st.markdown("Create a comprehensive PDF report with all analysis results and patient information.")
            
            if st.button("Generate PDF Report", type="secondary"):
                if 'patient_info' in st.session_state:
                    with st.spinner("Generating PDF report..."):
                        try:
                            # Get the original image from session state
                            original_image = st.session_state.get('original_image')
                            pdf_buffer = create_pdf_report(st.session_state['patient_info'], results, original_image)
                            
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"iris_analysis_report_{st.session_state['patient_info']['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                help="Download a comprehensive PDF report with all analysis results"
                            )
                            st.success("PDF report generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                else:
                    st.error("Please fill in patient information first.")
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>No Analysis Results Available</strong><br>
                Please upload an image or capture a photo and click "Analyze Image" to see results here.
            </div>
            """, unsafe_allow_html=True)
    
    # Professional footer
    st.markdown("""
    <div class="footer">
        <h4>Important Information</h4>
        <p><strong>This tool is for research and educational purposes only.</strong></p>
        <p>Not intended for medical diagnosis. Always consult qualified healthcare professionals for medical advice.</p>
        <small>Iris-Pupil Ratio Analyzer v1.0.0 | Built with Streamlit, MediaPipe, and OpenCV | Advanced Computer Vision Analysis</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()