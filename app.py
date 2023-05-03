import oci
import io
import json
import gradio as gr

from oci.object_storage import ObjectStorageClient
from oci.ai_language import AIServiceLanguageClient
from oci.ai_language.models import DetectLanguageKeyPhrasesDetails
from oci.ai_language.models import DetectLanguageSentimentsDetails

from oci.ai_speech import AIServiceSpeechClient
from oci.ai_speech.models import TranscriptionModelDetails
from oci.ai_speech.models import ObjectLocation
from oci.ai_speech.models import ObjectListInlineInputLocation
from oci.ai_speech.models import OutputLocation
from oci.ai_speech.models import CreateTranscriptionJobDetails

import base64
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from io import BytesIO

from oci.ai_vision import AIServiceVisionClient
from oci.ai_vision.models import AnalyzeImageDetails
from oci.ai_vision.models import ImageObjectDetectionFeature
from oci.ai_vision.models import ImageTextDetectionFeature
from oci.ai_vision.models import InlineImageDetails



###############################
###############################
############################### Functions
###############################
###############################
###############################
###############################

def preprocess(image_input):
    
    # Read in PIL Image - BytesIO Minics a File
    buffered = BytesIO()
    image_input.save(buffered, format="JPEG")
    
    # Enocde Image
    encoded_string = base64.b64encode(buffered.getvalue())
    
    return encoded_string


def detect_cars(config, encoded_string):
    
    # Max Result to return
    MAX_RESULTS = 100

    # Vision Service endpoint
    endpoint = "https://vision.aiservice.eu-frankfurt-1.oci.oraclecloud.com"

    # Initialize client service_endpoint is optional if it's specified in config
    ai_service_vision_client = AIServiceVisionClient(config=config, service_endpoint=endpoint)
    
    # Set up request body with one or multiple Features (Type of Service)
    image_object_detection_feature = ImageObjectDetectionFeature()
    image_object_detection_feature.max_results = MAX_RESULTS
    #image_object_detection_feature.model_id = "ocid1.aivisionmodel.oc1.eu-frankfurt-1.xx" ## <<<< Custom MODEL OCID"

    # List of Features
    features = [image_object_detection_feature]

    # Create Analyze Image Object and set Image and Features
    analyze_image_details = AnalyzeImageDetails()
    inline_image_details = InlineImageDetails()
    inline_image_details.data = encoded_string.decode('utf-8')
    analyze_image_details.image = inline_image_details
    analyze_image_details.features = features

    # Send analyze image request
    res = ai_service_vision_client.analyze_image(analyze_image_details=analyze_image_details)

    # Parse Response as JSON
    od_results = json.loads(str(res.data))
    
    return od_results


def parse_results(image_input, od_results):

    # Create Empty DataFrame
    results_df = pd.DataFrame([], columns = ["Label", "Confidence"])
    
    # Extract Bounding Boxes
    od_bounding_boxes = od_results['image_objects']
    
    # Convert PIL to Numpy array - CV2
    cv_image = np.array(image_input) 
    
    # Convert RGB to BGR - Reshape
    im = cv_image[:, :, ::-1].copy()
    
    # Fix colour
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Get Dimensions of Image
    height, width, channels = im.shape


    # Iterate over each Bounding Box
    for box in od_bounding_boxes:

        # Extract opposite coordinates for bounding box
        # Un-Normalise the Data by scaling to the max image height and width
        # Convert to Integer
        coordinates_pt1_x = int(box['bounding_polygon']['normalized_vertices'][0]['x'] * width)
        coordinates_pt1_y = int(box['bounding_polygon']['normalized_vertices'][0]['y'] * height)
        coordinates_pt2_x = int(box['bounding_polygon']['normalized_vertices'][2]['x'] * width)
        coordinates_pt2_y = int(box['bounding_polygon']['normalized_vertices'][2]['y'] * height)

        # Build Points as Tuples
        coordinates_pt1 = (coordinates_pt1_x, coordinates_pt1_y)
        coordinates_pt2 = (coordinates_pt2_x, coordinates_pt2_y)

        # Draw Bounding Boxes - Pass in Image, Top Left and Bottom Right Points, Colour, Line Thickness
        cv2.rectangle(im, coordinates_pt1, coordinates_pt2, (0, 255, 0), 2)

        # Plot Label just above the Top Left Point, Set Font, Size, Colour, Thickness
        cv2.putText(im, box['name'], (coordinates_pt1_x, coordinates_pt1_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        # Write Image with Bounding Boxes to file
        cv2.imwrite("./object_detection_result.png",im)

        # Extract Label and Confidence
        results_list = [box['name'], box['confidence']]

        # Append Results to DataFrame
        a_series = pd.Series(results_list, index = results_df.columns)
        results_df = results_df.append(a_series, ignore_index=True)

    return im, results_df


###############################
###############################
############################### Flow
###############################
###############################
###############################
###############################


def predict_image(image_input):
              
    #config = oci.config.from_file('/home/datascience/.oci/config', 'DEFAULT')  
    
    config = {'log_requests': False,
     'additional_user_agent': '',
     'pass_phrase': None,
     'user': 'ocid1.user.oc1..aaaaaaaakmn2nsfq47lzedlawxwk5a7uzgns2ek5uh2nhvbt2r67ek57tfea',
     'fingerprint': 'b7:d0:f5:1d:3c:10:a9:17:cd:04:2f:9f:f8:ed:8d:3f',
     'tenancy': 'ocid1.tenancy.oc1..aaaaaaaabu5fgingcjq3vc7djuwsdcutdxs4gsws6h4kfoldqpjuggxprgoa',
     'region': 'eu-frankfurt-1',
     'key_file': '/home/opc/tech_summit/gradio_tech_summit_2023/private_key.pem'}
    
    # Encoded PIL Image Recieved from Application
    encoded_string = preprocess(image_input)
        
    # Run Object Detection - Get Results
    od_results = detect_cars(config, encoded_string)
    
    # Parse Results and Draw Bounding Boxes
    image_output, results_df = parse_results(image_input, od_results)

    
    return image_output, results_df



###############################
###############################
############################### Gradio
###############################
###############################
###############################
###############################

with gr.Blocks() as demo:

    gr.Markdown("AI Vision, AI Speech, and AI Language")
   
    with gr.Tab("AI Vision"):
        
        #trigger
        button_1 = gr.Button("Analyse Image")
        
        ##input
        image_input = gr.Image(type='pil', label='Input Image')
        
        ##output
        with gr.Row():
            image_output = gr.Image(label='Output Image')
            results_df = gr.Dataframe(label = "Tabular results")
        
        
#     with gr.Tab("AI Speech & AI Language"):
#         ##input
#         input_recording = gr.Audio(type='filepath', label='Input Audio')
                
#         #trigger
#         button_2 = gr.Button("Analyse Recording")
        
#         ##output
#         transcription = gr.Text(label='Transcription')
#         key_phrases = gr.Text(label='Key Phrases')
#         sentiment = gr.Text(label='Sentiment')
#         neg_aspects = gr.Text(label='Negative Aspects Detected')


        ## buttons       
        button_1.click(predict_image, inputs=[image_input], outputs=[image_output, results_df])
        #button_2.click(predict_speech, inputs=[input_recording], outputs=[transcription, key_phrases, sentiment, neg_aspects])
            
demo.launch(debug=True, server_name="0.0.0.0") #width=800, height=1100, 