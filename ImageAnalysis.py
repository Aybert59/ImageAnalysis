# create a tkinter window
import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import torch
import torchvision
from torchvision.models import list_models, get_model
from GraphicStuff import create_window, display_log
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
#import torchvision.transforms as transforms
import numpy
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from huggingface_hub import HfApi

ImageZoneWidth = 1000
ImageZoneHeight = 1000
ColonneWidth = 250

# global variables
cap = None
detection = None
detection_button = None
device_button = None
Limage = None
DetectionThreshold = None
LastTime = cv2.getTickCount()
Fps = 0
FpsDisplay = 0

# from pytorch hub : 1: object detection 2: image classification 3: image segmentation
# from HuggingFace : 4: object detection 5: image classification 6: image segmentation
ModelType = 1 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9, device=device)
model.eval()
# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

def get_categories(listbox):
    # get the selected categories from the listbox
    categories = listbox.curselection()
    return categories

# function to display the image in the canvas  
def display_image(canvas, image):
    global Limage

    # resize the image to fit the canvas and keep aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        image = cv2.resize(image, (ImageZoneWidth, int(ImageZoneWidth / aspect_ratio)), interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(image, (int(ImageZoneHeight * aspect_ratio), ImageZoneHeight), interpolation=cv2.INTER_AREA)

    # convert the image to a PhotoImage
    photo = ImageTk.PhotoImage(image=Image.fromarray(image))

    # clear the canvas
    canvas.delete("all")
    # display the image in the canvas and center it 
    canvas.create_image(ImageZoneWidth // 2, ImageZoneHeight // 2, anchor="c", image=photo)
    canvas.image = photo

def do_detection (image, threshold, categories):
    global model
    global preprocess
    global ModelType

    img_tensor = pil_to_tensor(Image.fromarray(image))
    

    if ModelType == 1: # object detection
        batch = [preprocess(img_tensor)]
        prediction = model(batch)[0]
        cTime = cv2.getTickCount()

        #delete boxes with low score
        prediction["boxes"] = prediction["boxes"][prediction["scores"] > threshold]
        prediction["labels"] = prediction["labels"][prediction["scores"] > threshold]
        prediction["scores"] = prediction["scores"][prediction["scores"] > threshold]

        if (len(categories) > 0):
            #delete boxes with labels not in the list
            prediction["boxes"] = prediction["boxes"][torch.isin(prediction["labels"], torch.tensor(categories))] 
            prediction["scores"] = prediction["scores"][torch.isin(prediction["labels"], torch.tensor(categories))]
            prediction["labels"] = prediction["labels"][torch.isin(prediction["labels"], torch.tensor(categories))]

        display_log("Found {} objects :".format(len(prediction["labels"])),'info')

        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        for label, score in zip(labels, prediction["scores"]):
            display_log(label+" {:.2f}".format(score.item()*100)+"%", 'info') 

        box = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"],
                    labels=labels, colors="red", width=4, font = "Courier.ttc", font_size=60)
        im = to_pil_image(box.detach())
        return cTime, numpy.array(im)
    
    elif ModelType == 2: # image segmentation
        batch = preprocess(img_tensor).unsqueeze(0)
        
        prediction = model(batch)["out"]
        normalized_masks = prediction.softmax(dim=1)
        cTime = cv2.getTickCount()

        #resize the normalized_masks tensor like img_tensor
        masks = torch.nn.functional.interpolate(normalized_masks, size=(img_tensor.shape[1], img_tensor.shape[2]), mode="bilinear")

        class_dim = 1
        if (len(categories) > 0):
            #keep only masks within categories list
            mask = torch.isin(masks.argmax(class_dim), torch.tensor(categories))
        else: 
            mask = (masks.argmax(class_dim) == 0) # keep only the background

        # Step 6: Visualize the predicted masks on the image, color green   

        box = draw_segmentation_masks(img_tensor, mask, alpha=0.5, colors="green")
        im = to_pil_image(box.detach())
        return cTime, numpy.array(im)
        
    
    elif ModelType == 3: # image classification
        batch = preprocess(img_tensor).to(device).unsqueeze(0)
        
        # Step 4: Use the model and print the predicted category
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        cTime = cv2.getTickCount()

        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        display_log(f"{category_name}: {100 * score:.1f}%",'info')
        #write the label on the image
        im = image.copy()
        im = cv2.putText(im, f"{category_name}: {100 * score:.1f}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
        return cTime, im
    
    elif ModelType == 4: # image classification from HuggingFace
        
        #resize the image
        input_image = cv2.resize(image, (model.config.image_size, model.config.image_size))
        batch = preprocess (input_image, return_tensors='pt').to(device)

        with torch.no_grad():
            prediction = model(**batch)

        class_id = prediction.logits.argmax().item()
        score = prediction.logits.softmax(1)[0][class_id].item()
        cTime = cv2.getTickCount()

        category_name = model.config.id2label[class_id]

        display_log(f"{category_name}: {100 * score:.1f}%",'info')
        #write the label on the image
        im = image.copy()
        im = cv2.putText(im, f"{category_name}: {100 * score:.1f}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
        return cTime, im
    
    elif ModelType == 5: # image segmentation from HuggingFace

        #inputs = preprocess(images=image, return_tensors="pt").to(device)
        inputs = preprocess(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)
        cTime = cv2.getTickCount()

        # resize output to match input image dimensions
        upsampled_logits = torch.nn.functional.interpolate(logits,
                size=(image.shape[0], image.shape[1]), # H x W
                mode='bilinear',
                align_corners=False)
        

        #########  et rechercher comment mettre couleurs différentes
        class_dim = 1
        if (len(categories) > 0):
            #keep only masks within categories list
            predicted_mask = torch.isin(upsampled_logits.argmax(class_dim), torch.tensor(categories))
        else: 
            predicted_mask = (upsampled_logits.argmax(class_dim) == 0) # keep only the background

        # Step 6: Visualize the predicted masks on the image, color green
        box = draw_segmentation_masks(img_tensor, predicted_mask.cpu().bool(), alpha=0.5, colors="green")
        im = to_pil_image(box.detach())
    
        return cTime, numpy.array(im)
    
    elif ModelType == 6: # object detection from HuggingFace
        
        #resize the image
        batch = preprocess (images=image, return_tensors='pt').to(device)

        with torch.no_grad():
            prediction = model(**batch)
        cTime = cv2.getTickCount()

        #logits = prediction.logits
        #bboxes = prediction.pred_boxes

        target_sizes = torch.tensor([image.shape[:2]])
        results = preprocess.post_process_object_detection(prediction, threshold=threshold, target_sizes=target_sizes)[0]

        if (len(categories) > 0):
            #delete boxes with labels not in the list
            results["boxes"] = results["boxes"][torch.isin(results["labels"], torch.tensor(categories))] 
            results["scores"] = results["scores"][torch.isin(results["labels"], torch.tensor(categories))]
            results["labels"] = results["labels"][torch.isin(results["labels"], torch.tensor(categories))]

        im = image.copy()
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

            im = cv2.putText(im, f"{model.config.id2label[label.item()]}: {100 * score:.1f}%", (int(box[0]), int(box[1])+50), 
                             cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)
            im = cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        
        return cTime, im
    
def display_camera_image(canvas, labelfps, listbox):
    global cap
    global Limage
    global detection
    global DetectionThreshold
    global Fps
    global FpsDisplay
    global LastTime

    cats = get_categories(listbox)

    if cap is not None: 
        ret, image = cap.read()
        Limage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if detection is not None:
            cTime, im = do_detection (Limage, DetectionThreshold.get()/100.0, cats)
            display_image(canvas, im)
        else:
            display_image(canvas, Limage)

        newTime = cv2.getTickCount()
        Fps += 1
        t = (newTime - LastTime)/cv2.getTickFrequency()
        if t > 1.0:
                LastTime = newTime
                FpsDisplay = Fps
                Fps = 0
        labelfps.config(text=str(int(FpsDisplay)) + " fps")
        window.after(10, display_camera_image, canvas, labelfps, listbox)

def activate_camera(canvas, label, labelfps, listbox):
    global cap

    if cap is None:
        label.config(text="Camera activated")
        cap = cv2.VideoCapture(0)
        display_camera_image(canvas, labelfps, listbox)

def deactivate_camera(canvas, label):
    global cap
    global Limage

    label.config(text="")
    if cap is not None:
        cap.release()
    cap = None

    canvas.delete("all")
    canvas.image = None
    Limage = None

def activate_detection(canvas, labelfps, listbox):
    global detection
    global detection_button
    global Limage
    global DetectionThreshold

    cats = get_categories(listbox)
    
    if detection is None and Limage is not None:
        detection_button.config(text="Detection On")
        
        sTime = cv2.getTickCount()
        cTime, im = do_detection (Limage, DetectionThreshold.get()/100.0, cats)
        eTime = cv2.getTickCount()
        d = (cTime - sTime)/cv2.getTickFrequency()
        d2 = (eTime - sTime)/cv2.getTickFrequency()

        display_image(canvas, im)
        labelfps.config(text=str(float("{:.3f}".format(d))) + " sec")
        display_log("Detection + drawing time : " + str(float("{:.3f}".format(d2))) + " sec", "info") 
        detection = True
    else:
        detection_button.config(text="Detection Off")
        detection = None
        if Limage is not None:
            canvas.delete("all")
            display_image(canvas, Limage)
            labelfps.config(text="-- fps")


def select_file(canvas, label, labelfps):
    global Limage

    deactivate_camera(canvas, label)
    labelfps.config(text="-- fps")
   
    file_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~/Pictures"), filetypes=(("Image Files", ".jpg .jpeg .png")))

    # check if a file was selected
    if file_path:
        
        image = cv2.imread(file_path)
        if image is not None:
            label.config(text=file_path.split("/")[-1])
            # display the image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Limage = image
            display_image(canvas, image)
        else:
            messagebox.showerror("Error", "Failed to load image.")

def load_hub_model (canvas, label, model_name, labelfps, listbox, device_button):
    global model
    global DetectionThreshold
    global preprocess
    global weights
    global device

    if detection is not None:
        activate_detection(canvas, labelfps, listbox)

    label.config(text=model_name)
    

    #pour info seulement
    weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name=model_name)
    weights = weight_enum['DEFAULT']
  
    #model = get_model(model_name, weights=weights, box_score_thresh=DetectionThreshold.get()/100.0, device=device)  
    try: 
        model = get_model(model_name, weights=weights, device=device)
        device_button.config(text=f"Device:{device.upper()}")
    except:
        model = get_model(model_name, weights=weights)
        device_button.config(text="Device:CPU")
        device="cpu"
    
    model.eval()
    preprocess = weights.transforms()

    listbox.delete(0, tk.END)
    listbox.insert(0, *weights.meta["categories"])  

def select_OD_model(canvas, menu, label, model_name, labelfps, window, listbox, label_dev):
    global ModelType

    load_hub_model (canvas, label, model_name, labelfps, listbox, label_dev)
    window.title("Object Detection")
    ModelType = 1


def select_SEG_model(canvas,  menu, label, model_name, labelfps, window, listbox, label_dev):
    global ModelType

    load_hub_model (canvas, label, model_name, labelfps, listbox, label_dev)
    window.title("Image Segmentation")
    ModelType = 2
    

def select_CL_model(canvas,  menu, label, model_name, labelfps, window, listbox, label_dev):
    global ModelType

    load_hub_model (canvas, label, model_name, labelfps, listbox, label_dev)
    window.title("Image Classification")
    ModelType = 3

def select_hf_CL_model(canvas,  menu, label, labelfps, window, listbox, HF_name):
    global ModelType
    global preprocess
    global model
    
    if HF_name is  None:
        HF_name = simpledialog.askstring("Model", "Classification model ?")

    if HF_name is not None:
        preprocess = AutoImageProcessor.from_pretrained(HF_name)
        model = AutoModelForImageClassification.from_pretrained(HF_name)
        model.to(device)
        model.eval()

        label.config(text=HF_name)

        listbox.delete(0, tk.END)
        listbox.insert(0, *model.config.id2label.values())

        window.title("HuggingFace Image Classification")
        ModelType = 4

def select_hf_SEG_model(canvas,  menu, label, labelfps, window, listbox, HF_name):
    global ModelType
    global preprocess
    global model
    
    if HF_name is  None:
        HF_name = simpledialog.askstring("Model", "Segmentation model ?")

    if HF_name is not None:
        preprocess = SegformerImageProcessor.from_pretrained(HF_name)
        model = SegformerForSemanticSegmentation.from_pretrained(HF_name)
        model.to(device)
        model.eval()

        label.config(text=HF_name)

        listbox.delete(0, tk.END)
        listbox.insert(0, *model.config.id2label.values())

        window.title("HuggingFace Image Segmentation")
        ModelType = 5

def select_hf_OD_model(canvas,  menu, label, labelfps, window, listbox, HF_name):
    global ModelType
    global preprocess
    global model
    
    if HF_name is  None:
        HF_name = simpledialog.askstring("Model", "Object Detection model ?")

    if HF_name is not None:
        preprocess = DetrFeatureExtractor.from_pretrained(HF_name)
        model = DetrForObjectDetection.from_pretrained(HF_name)
        model.to(device)
        model.eval()

        label.config(text=HF_name)

        listbox.delete(0, tk.END)
        listbox.insert(0, *model.config.id2label.values())

        window.title("HuggingFace Object Detection")
        ModelType = 6

def select_local_CL_model(canvas,  menu, label, labelfps, window, listbox, HF_name):
    global ModelType
    global preprocess
    global model

    file_path = filedialog.askopenfilename(initialdir=os.path.expanduser("~"), title="Select a model")
    file_path = os.path.dirname(file_path)

    if file_path is not None:
        preprocess = AutoImageProcessor.from_pretrained(file_path)
        model = AutoModelForImageClassification.from_pretrained(file_path)
        model.to(device)
        model.eval()

        label.config(text=HF_name)

        listbox.delete(0, tk.END)
        listbox.insert(0, *model.config.id2label.values())

        window.title("Local Image Classification")
        ModelType = 4
    

def select_local_SEG_model(canvas,  menu, label, labelfps, window, listbox, HF_name):
    global ModelType
    global preprocess
    global model
    
    return

def select_local_OD_model(canvas,  menu, label, labelfps, window, listbox, HF_name):
    global ModelType
    global preprocess
    global model
    
    return

def toggle_device():
    global device
    global model
    
    if device == "cpu":
        device = "mps"
    else:
        device = "cpu"

    device_button.config(text=f"Device:{device.upper()}")
    model.to(device)
    
    

window, canvas, widget_bar, menubar = create_window (ImageZoneWidth, ImageZoneHeight, ColonneWidth)
DetectionThreshold = tk.IntVar()

#create a label to display the image name
label_image = tk.Label(widget_bar, text="No image selected")
#create a label to display the model name
label_model = tk.Label(widget_bar, text="Default: FasterRCNN")
#create a label to display the fps rate
label_fps = tk.Label(widget_bar, text="-- fps")
#create an empty listbox to display the detection categories
listbox = tk.Listbox(widget_bar, selectmode=tk.MULTIPLE, exportselection=False)
listbox.insert(0, *weights.meta["categories"])

#create a button to activate camera
cam_button = tk.Button(widget_bar, text="Camera", command=lambda arg1=canvas, arg2=label_image, arg3=label_fps, arg4=listbox : 
                       activate_camera(arg1, arg2, arg3, arg4))
#create a button to select a file
img_button = tk.Button(widget_bar, text="Select image", command=lambda arg1=canvas, arg2=label_image, arg3=label_fps : 
                       select_file(arg1, arg2, arg3))
#create a button to toggle detection
detection_button = tk.Button(widget_bar, text="Detection Off", command=lambda arg1=canvas, arg3=label_fps, arg4=listbox : 
                        activate_detection(arg1, arg3, arg4))
#create a button to toggle device
device_button = tk.Button(widget_bar, text=f"Device: {device.upper()}", command=lambda  : toggle_device())

#create a slider to adjust the detection threshold
slider = tk.Scale(widget_bar, from_=0, to=100, orient=tk.HORIZONTAL, label="Threshold", variable=DetectionThreshold)
slider.set(90)

window.title("Object Detection")  # default model

cam_button.pack(side=tk.TOP, fill=tk.X)
tk.Label(widget_bar, text="").pack(side=tk.TOP, fill=tk.X) # space
img_button.pack(side=tk.TOP, fill=tk.X)
label_image.pack(side=tk.TOP, fill=tk.X)
tk.Label(widget_bar, text="").pack(side=tk.TOP, fill=tk.X) # space
detection_button.pack(side=tk.TOP, fill=tk.X)
label_model.pack(side=tk.TOP, fill=tk.X)
listbox.pack(side=tk.TOP, fill=tk.X, ipadx=10)
tk.Label(widget_bar, text="").pack(ipady=30, side=tk.TOP, fill=tk.X) # space
slider.pack(side=tk.TOP, fill=tk.X)
tk.Label(widget_bar, text="").pack(ipady=30, side=tk.TOP, fill=tk.X) # space
device_button.pack(side=tk.TOP, fill=tk.X)
label_fps.pack(side=tk.BOTTOM, fill=tk.X)


#create a menu to select an object detection model
OD_models = list_models(module=torchvision.models.detection)
OD_menu = tk.Menu(menubar, tearoff=0)
for model_name in OD_models:
    OD_menu.add_command(label=model_name, command=lambda model_name=model_name, arg3=label_fps, arg4=window, arg5=listbox, arg6=device_button : 
                        select_OD_model(canvas, menubar, label_model, model_name, arg3, arg4, arg5, arg6))
menubar.add_cascade(label="Object Detection", menu=OD_menu)

#create a menu to select a segmentation model
SEG_models = list_models(module=torchvision.models.segmentation)
SEG_menu = tk.Menu(menubar, tearoff=0)
for model_name in SEG_models:
    SEG_menu.add_command(label=model_name, command=lambda model_name=model_name, arg3=label_fps, arg4=window, arg5=listbox, arg6=device_button : 
                         select_SEG_model(canvas, menubar, label_model, model_name, arg3, arg4, arg5, arg6))
menubar.add_cascade(label="Image Segmentation", menu=SEG_menu)

#create a menu to select a classification model
CL_models = list_models(module=torchvision.models)
CL_menu = tk.Menu(menubar, tearoff=0)
for model_name in CL_models:
    CL_menu.add_command(label=model_name, command=lambda model_name=model_name, arg3=label_fps, arg4=window, arg5=listbox, arg6=device_button : 
                        select_CL_model(canvas, menubar, label_model, model_name, arg3, arg4, arg5, arg6))
menubar.add_cascade(label="Image Classification", menu=CL_menu)

#create a menu to select external model
HF_menu = tk.Menu(menubar, tearoff=0)
HF_menu.add_command(label='Classification', command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_hf_CL_model(canvas, menubar, label_model,  arg3, arg4, arg5, None))
HF_menu.add_command(label='Segmentation', command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_hf_SEG_model(canvas, menubar, label_model,  arg3, arg4, arg5, None))
HF_menu.add_command(label='Object Detection', command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_hf_OD_model(canvas, menubar, label_model,  arg3, arg4, arg5, None))
HF_menu.add_separator()


#if file "favorites" exists open it
if os.path.isfile("favorites"):
    with open("favorites", "r") as f:
        for line in f:

            #split line by semicolumn
            model_name, model_type, model_comment = line.split(";")

            if model_type == "OD":
                HF_menu.add_command(label=model_name, command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_hf_OD_model(canvas, menubar, label_model,  arg3, arg4, arg5, model_name))
            elif model_type == "SEG":
                HF_menu.add_command(label=model_name, command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_hf_SEG_model(canvas, menubar, label_model,  arg3, arg4, arg5, model_name))
            elif model_type == "CL":
                HF_menu.add_command(label=model_name, command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_hf_CL_model(canvas, menubar, label_model,  arg3, arg4, arg5, model_name))

menubar.add_cascade(label="HuggingFace Models", menu=HF_menu)

#create a menu to select external model
localmenu = tk.Menu(menubar, tearoff=0)
localmenu.add_command(label='Chose an object detection model...', command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_local_OD_model(canvas, menubar, label_model,  arg3, arg4, arg5, None))
localmenu.add_command(label='Chose a segmentation model...', command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_local_SEG_model(canvas, menubar, label_model,  arg3, arg4, arg5, None))
localmenu.add_command(label='Chose a classification model...', command=lambda arg3=label_fps, arg4=window, arg5=listbox : 
                        select_local_CL_model(canvas, menubar, label_model,  arg3, arg4, arg5, None))
menubar.add_cascade(label="Local models", menu=localmenu)

window.mainloop()


########
# a faire : un bouton qui tente : model = model.to(device) et qui affiche, si réussi



