import oci
import io
import json
import gradio as gr
import time

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
############################### Functions Speech
###############################
###############################
###############################
###############################


def post_audio_to_os(config, filename, audio):
    
    # Instantiate Object Storage Client
    oci_os_client = ObjectStorageClient(config)
    
    # Define Parameters
    bucket_name = "West_BP"
    namespace = "frqap2zhtzbe"
    
    # Push File to Bucket
    resp = oci_os_client.put_object(namespace, bucket_name, filename, io.open(audio, 'rb'), content_type='audio/wav')


def run_speech_model(config, filename):
    
    # Instantiate Speech Client
    ai_speech_client = AIServiceSpeechClient(config)
    
    # Define Parameters for Transcription Jobs
    job_display_name = "OCI-DS-Job-Demo"
    job_compartment_id = "ocid1.compartment.oc1..aaaaaaaae3n6r6hrjipbap2hojicrsvkzatrtlwvsyrpyjd7wjnw4za3m75q"
    job_description = "Shop Complaint"
    bucket_name = "West_BP"
    namespace = "frqap2zhtzbe"
    output_prefix = "Speech-Output"
    

    # Define Transcription Job - Model, Data, Input, Outputs
    job_model_details = TranscriptionModelDetails(domain="GENERIC", language_code="en-GB")
    job_object_location = ObjectLocation(namespace_name=namespace, bucket_name=bucket_name,object_names=[filename])
    job_input_location = ObjectListInlineInputLocation(location_type="OBJECT_LIST_INLINE_INPUT_LOCATION", object_locations=[job_object_location])
    job_output_location = OutputLocation(namespace_name=namespace, bucket_name=bucket_name, prefix=output_prefix)

    
    # Create Transcription Job with details provided above
    transcription_job_details = CreateTranscriptionJobDetails(display_name=job_display_name,
                                                                compartment_id=job_compartment_id,
                                                                description=job_description,
                                                                model_details=job_model_details,
                                                                input_location=job_input_location,
                                                                output_location=job_output_location)

    
    
    # Call the AI Speech Service to Create Transcription Job 
    transcription_job = None
    try:
        transcription_job = ai_speech_client.create_transcription_job(create_transcription_job_details=transcription_job_details)
    except Exception as e:
        print(e)
    else:
        print(transcription_job.data.lifecycle_state)
        
    
    # Pause for 5 Seconds to Allow Job to be Accepted
    time.sleep(5)
    
    
    # Gets the First Transcription Tasks under given Transcription Job Id then Extracts Info for that Task
    transcription_tasks = None
    try:
        # Get Tasks Under Job
        transcription_tasks = ai_speech_client.list_transcription_tasks(transcription_job.data.id, limit=1)
        
        # Keep Checking until Task is Succeeded
        while transcription_tasks.data.items[0].lifecycle_state != 'SUCCEEDED':
            print('Transcribing in Progress...')
            time.sleep(5)
            transcription_tasks = ai_speech_client.list_transcription_tasks(transcription_job.data.id, limit=1)
            
        # Once Task is Succeeded Extract Task Info
        transcription_task = ai_speech_client.get_transcription_task(transcription_job.data.id, transcription_tasks.data.items[0].id)
        
    except Exception as e:
        print(e)
        
    else:
        print(transcription_tasks.data.items[0].lifecycle_state)
        print(transcription_task.data.output_location.object_names[0])
        
    
    # Extract Results File Name from Task Info Response
    object_name = transcription_task.data.output_location.object_names[0]
    
    
    return object_name


def parse_results(config, object_name):
    
    # Instantiate Object Storage Client
    oci_os_client = ObjectStorageClient(config)
    
    # Define Parameters
    bucket_name = "West_BP"
    namespace = "frqap2zhtzbe"
    
    # Get Speech Results File from Object Storage
    resp = oci_os_client.get_object(namespace, bucket_name, object_name)
    
    # Decode Results from File
    decoded_resp = json.loads(resp.data.content.decode())
    
    # Extract Transcription from Results
    transcription = decoded_resp['transcriptions'][0]['transcription']
    
    return transcription


def run_language_models(config, transcription):
    
    # Initialize Service Client to Language API
    ai_language_client = AIServiceLanguageClient(config)
    
    
    # Make a REST API Request to AI Language Service to Detect Key Phrases
    language_key_phrases = ai_language_client.detect_language_key_phrases(
        detect_language_key_phrases_details=DetectLanguageKeyPhrasesDetails(text = transcription))
    
    # Results List
    key_phrase_results = []
    
    # Extract Language Entities
    formatted_response = language_key_phrases.data.key_phrases
    
    # Iterate through and Store Entites in Results List
    for key_phrase in formatted_response:
        key_phrase_results.append(key_phrase.text)
    
    
    
    # Make a REST API Request to AI Language Service to Detect Sentiments
    language_sentiment_response = ai_language_client.detect_language_sentiments(
        detect_language_sentiments_details=DetectLanguageSentimentsDetails(text = transcription))
    
    # Results List
    sentiment_results = []
    
    # Extract Language Sentiments
    formatted_response = language_sentiment_response.data.aspects
    
    # Iterate through and Store Aspect Sentiment in Results List
    for aspect in formatted_response:
        sentiment_results.append((aspect.text, aspect.sentiment))

    
    return key_phrase_results, sentiment_results



    
###############################
###############################
############################### Flow Speech
###############################
###############################
###############################
###############################

def predict_speech(audio):
    
    # Authenticate against OCI
    config = {'log_requests': False,
     'additional_user_agent': '',
     'pass_phrase': None,
     'user': 'ocid1.user.oc1..aaaaaaaakmn2nsfq47lzedlawxwk5a7uzgns2ek5uh2nhvbt2r67ek57tfea',
     'fingerprint': 'b7:d0:f5:1d:3c:10:a9:17:cd:04:2f:9f:f8:ed:8d:3f',
     'tenancy': 'ocid1.tenancy.oc1..aaaaaaaabu5fgingcjq3vc7djuwsdcutdxs4gsws6h4kfoldqpjuggxprgoa',
     'region': 'eu-frankfurt-1',
     'key_file': '/home/opc/tech_summit/gradio_tech_summit_2023/private_key.pem'}

    
    # Send Audio File to Object storage
    filename = 'complaint-example.wav'
    post_audio_to_os(config, filename, audio)
    
    # Run Speech Model - Returns Results File Name
    object_name = run_speech_model(config, filename)
    
    # Get Results File from Object Storage and Parse Transcription
    transcription = parse_results(config, object_name)
    
    # Run Language Models on Transcription to Get Key Phrases and Sentiment
    key_phrases, sentiment = run_language_models(config, transcription)
    
    # Count Negative Aspects
    neg_aspects = 0 
    
    for sen in sentiment:
        if sen[1] == 'Negative':
            neg_aspects += 1
    
    
    return transcription, key_phrases, sentiment, neg_aspects




















###############################
###############################
############################### Functions Image
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


def parse_results_image(image_input, od_results):

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
        #results_df = results_df.append(a_series, ignore_index=True)

        results_df = pd.concat([results_df, pd.DataFrame([a_series])], ignore_index=True)

    return im, results_df


###############################
###############################
############################### Functions Image Lego City
###############################
###############################
###############################
###############################

def preprocess_lego(image_input_lego):
    
    # Read in PIL Image - BytesIO Minics a File
    buffered = BytesIO()
    image_input_lego.save(buffered, format="JPEG")
    
    # Enocde Image
    encoded_string_lego = base64.b64encode(buffered.getvalue())
    
    return encoded_string_lego


def detect_cars_lego(config, encoded_string_lego):
    
    # Max Result to return
    MAX_RESULTS = 15

    # Vision Service endpoint
    endpoint = "https://vision.aiservice.eu-frankfurt-1.oci.oraclecloud.com"

    # Initialize client service_endpoint is optional if it's specified in config
    ai_service_vision_client = AIServiceVisionClient(config=config, service_endpoint=endpoint)
    
    # Set up request body with one or multiple Features (Type of Service)
    image_object_detection_feature = ImageObjectDetectionFeature()
    image_object_detection_feature.max_results = MAX_RESULTS
    image_object_detection_feature.model_id = "ocid1.aivisionmodel.oc1.eu-frankfurt-1.amaaaaaangencdyafgnt65gu7hgbiaxoi6puvgouysn6ranbd3zh3p3qmoda" ## <<<< Custom MODEL OCID"

    # List of Features
    features = [image_object_detection_feature]

    # Create Analyze Image Object and set Image and Features
    analyze_image_details = AnalyzeImageDetails()
    inline_image_details = InlineImageDetails()
    inline_image_details.data = encoded_string_lego.decode('utf-8')
    analyze_image_details.image = inline_image_details
    analyze_image_details.features = features

    # Send analyze image request
    res = ai_service_vision_client.analyze_image(analyze_image_details=analyze_image_details)

    # Parse Response as JSON
    od_results_lego = json.loads(str(res.data))
    
    return od_results_lego


def parse_results_image_lego(image_input_lego, od_results_lego):

    # Create Empty DataFrame
    results_df_lego = pd.DataFrame([], columns = ["Label", "Confidence"])
    
    # Extract Bounding Boxes
    od_bounding_boxes = od_results_lego['image_objects']
    
    # Convert PIL to Numpy array - CV2
    cv_image = np.array(image_input_lego) 
    
    # Convert RGB to BGR - Reshape
    im = cv_image[:, :, ::-1].copy()
    
    # Fix colour
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Get Dimensions of Image
    height, width, channels = im.shape


    # Iterate over each Bounding Box
    for box in od_bounding_boxes:
        
        if box['confidence'] >= 0.5:

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
            cv2.imwrite("./object_detection_result_lego.png",im)
    
            # Extract Label and Confidence
            results_list_lego = [box['name'], box['confidence']]
    
            # Append Results to DataFrame
            a_series_lego = pd.Series(results_list_lego, index = results_df_lego.columns)
            #results_df = results_df.append(a_series, ignore_index=True)
    
            results_df_lego = pd.concat([results_df_lego, pd.DataFrame([a_series_lego])], ignore_index=True)

        else:
            # Write Image with No Bounding Boxes to file
            cv2.imwrite("./object_detection_result_lego.png",im)

    return im, results_df_lego


###############################
###############################
############################### Flow Image
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
    image_output, results_df = parse_results_image(image_input, od_results)

    
    return image_output, results_df



###############################
###############################
############################### Flow Image Lego
###############################
###############################
###############################
###############################


def predict_image_lego(image_input_lego):
              
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
    encoded_string_lego = preprocess_lego(image_input_lego)
        
    # Run Object Detection - Get Results
    od_results_lego = detect_cars_lego(config, encoded_string_lego)
    
    # Parse Results and Draw Bounding Boxes
    image_output_lego, results_df_lego = parse_results_image_lego(image_input_lego, od_results_lego)

    
    return image_output_lego, results_df_lego


###############################
###############################
############################### Gradio
###############################
###############################
###############################
###############################

with gr.Blocks() as demo:

    gr.Markdown("AI Vision, AI Speech, and AI Language")


    with gr.Tab("AI Vision - Lego City"):
        
        #trigger
        button_0 = gr.Button("Analyse Image - Lego City")
        
        ##input
        image_input_lego = gr.Image(type='pil', label='Input Image - Lego City')
        
        ##output
        with gr.Row():
            image_output_lego = gr.Image(label='Output Image')
            results_df_lego = gr.Dataframe(label = "Tabular results")
   
    with gr.Tab("AI Vision"):
        
        #trigger
        button_1 = gr.Button("Analyse Image")
        
        ##input
        image_input = gr.Image(type='pil', label='Input Image')
        
        ##output
        with gr.Row():
            image_output = gr.Image(label='Output Image')
            results_df = gr.Dataframe(label = "Tabular results")
        
        
    with gr.Tab("AI Speech & AI Language"):
        ##input
        input_recording = gr.Audio(type='filepath', label='Input Audio')
                
        #trigger
        button_2 = gr.Button("Analyse Recording")
        
        ##output
        transcription = gr.Text(label='Transcription')
        key_phrases = gr.Text(label='Key Phrases')
        sentiment = gr.Text(label='Sentiment')
        neg_aspects = gr.Text(label='Negative Aspects Detected')


        ## buttons       
        button_0.click(predict_image_lego, inputs=[image_input_lego], outputs=[image_output_lego, results_df_lego])
        button_1.click(predict_image, inputs=[image_input], outputs=[image_output, results_df])
        button_2.click(predict_speech, inputs=[input_recording], outputs=[transcription, key_phrases, sentiment, neg_aspects])
            
demo.launch(server_name="0.0.0.0") #width=800, height=1100, 
