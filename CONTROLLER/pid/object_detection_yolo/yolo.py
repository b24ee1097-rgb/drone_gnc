def load_yolo_model():
    """
    Load the pre-trained YOLO model.
    """
    try:
        # Load YOLOv8 model 
        model = YOLO("yolov8n.pt")  # nano model
        print("YOLO model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        return None