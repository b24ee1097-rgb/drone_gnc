#Step 1:
def initialize_midas():
   """Initialize the MiDaS model once"""
   global midas, transform
  
   if midas is None:
       # Load the MiDaS model
       midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
       # Set the model to evaluation mode
       midas.eval()
       # Load necessary image transformations
       transformss = torch.hub.load('intel-isl/MiDaS', 'transforms')
       transform = transformss.small_transform
      
   return midas, transform
#Step 2:
def create_depth_map(frame):
   """Create a depth map using already loaded MiDaS"""
   global midas, transform
  
   # Ensure MiDaS is initialized
   if midas is None or transform is None:
       midas, transform = initialize_midas()
  
   # Prepare the image
   image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   image_batch = transform(image)    
  
   # Disable gradient calculation for inference
   with torch.no_grad():
       # Get the prediction
       prediction = midas(image_batch)
      
       # Resize to original image size
       prediction = torch.nn.functional.interpolate(
           prediction.unsqueeze(1),
           size=image.shape[:2],
           mode='bicubic',
           align_corners=False
       ).squeeze()
      
       # Convert to numpy array
       output = prediction.numpy()
  
   # Normalize the depth map
   depth_map = (output - output.min()) / (output.max() - output.min())
  
   return depth_map
#step 3:
def get_scale_factor(depth_map, center_x, center_y, known_distance=10):
   """
   Compute the scale factor
   Args:
       depth_map: MiDaS depth map
       center_x, center_y: Coordinates of the clip center
       known_distance: Known distance of the clip to the camera (in centimeters)
  
   Returns:
       scale_factor: Scale factor to convert MiDaS values to centimeters
   """
   # Check if coordinates are valid
   if center_x is not None and center_y is not None:
       # Convert to integers
       center_x_int = int(center_x)
       center_y_int = int(center_y)
      
       # Check if coordinates are within depth_map bounds
       if (0 <= center_y_int < depth_map.shape[0]) and (0 <= center_x_int < depth_map.shape[1]):
           # Get MiDaS depth at the clip center
           clip_depth_value = depth_map[center_y_int, center_x_int]
          
           # Avoid division by zero
           if clip_depth_value > 0:
               # Calculate scale factor (real distance / MiDaS value)
               scale_factor = known_distance / clip_depth_value
               return scale_factor
  
   # Default value if calibration fails
   print("Calibration failed, using default scale factor")
   return 1.0  # Default scale factor
#step 4:
def calculate_distances(depth_map, scale_factor):
   """
   Calculate distances for all points in the image  
   Returns:
       distance_map: Distance map in centimeters
   """
  
   # Convert entire depth map to metric distances
   distance_map = depth_map * scale_factor
  
   return distance_map