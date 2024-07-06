import BaboonMMBPackage

# Define parameters
params = {'K': 4, 'CONNECTIVITY': 8, 'AREA_MIN': 5, 'AREA_MAX': 80, 'ASPECT_RATIO_MIN': 1.0,
          'ASPECT_RATIO_MAX': 6.0, 'L': 4, 'KERNEL': 3, 'BITWISE_OR': False, 'PIPELINE_LENGTH': 5,
          'PIPELINE_SIZE': 7, 'H': 3, 'MAX_NITER_PARAM': 10, 'GAMMA1_PARAM': 0.8, 'GAMMA2_PARAM': 0.8,
          'FRAME_RATE': 10, 'IMAGE_SEQUENCE': 'input/viso_video_1'}

# Call the MATLAB function and get the returned data
pkg = BaboonMMBPackage.initialize()
result = pkg.baboon_mmb(params)

# Extract objects from the result struct
objects = result['objects']

# Process the returned data
for obj in objects:
    print(f"Frame Number: {obj['frameNumber']}, ID: {obj['id']}, X: {obj['x']}, Y: {obj['y']}, Width: {obj['width']}, Height: {obj['height']}")

# Terminate the package
pkg.terminate()