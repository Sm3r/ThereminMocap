import pyzed.sl as sl

class Zed():
    def __init__(self,filename=None,depth_confidence=100):

        print("Bringing Up ZED CAMERA Information...")
        # Decide if SVO or Live
        if filename is None:
            print("Using Live stream from ZED camera")
            self.input_type = sl.InputType()
            self.svo_mode = False
        else:
            filepath = filename
            print("Reading SVO file: {0}".format(filepath))
            self.input_type = sl.InputType()
            self.input_type.set_from_svo_file(filepath)
            self.svo_mode = True

        # Initialize the ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(input_t=self.input_type)

        if self.svo_mode:
            # Best quality for dataset creation – speed doesn't matter
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
            self.init_params.svo_real_time_mode = False
            confidence = 20
        else:
            # Live preview – prioritise FPS
            self.init_params.camera_resolution = sl.RESOLUTION.VGA
            self.init_params.camera_fps = 30
            self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            
            confidence = depth_confidence

        self.init_params.depth_minimum_distance = 0.3
        self.init_params.depth_maximum_distance = 40
        self.init_params.coordinate_units = sl.UNIT.METER

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS :
            msg = repr(err)
            print(msg)
            print("If using SVO, check if the path is correct")
            try:
                self.zed.close()
            except Exception:
                pass
            raise RuntimeError(f"ZED open failed: {msg}")

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        #if self.svo_mode:
        #    self.runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
        self.runtime_parameters.confidence_threshold = confidence

        # Get Camera Calibration Parameters
        # self.camera_params = self.zed.get_camera_information().calibration_parameters.left_cam
        self.camera_params = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.fx = self.camera_params.fx  # Focal length in pixels (x-axis)
        self.fy = self.camera_params.fy  # Focal length in pixels (y-axis)
        self.cx = self.camera_params.cx  # X-coordinate of the principal point
        self.cy = self.camera_params.cy  # Y-coordinate of the principal point

        


        # declare image, depth, point cloud
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        self.confidence_map = sl.Mat()

    def print_information(self):
        print("Resolution: {0}, {1}.".format(
            self.zed.get_camera_information().camera_configuration.resolution.width,
            self.zed.get_camera_information().camera_configuration.resolution.height
        ))
        #print("Camera FPS: {0}".format(self.zed.get_camera_information().camera_fps))
        cam_info = self.zed.get_camera_information()
        fps = cam_info.camera_configuration.fps

        print("Camera FPS: {0}".format(fps))
        print("Depth mode: {0}.".format(self.init_params.depth_mode))
        #print("Sensing mode: {0}.".format(self.runtime_parameters.sensing_mode))
        if self.svo_mode:
            print("Frame count: {0}.\n".format(self.zed.get_svo_number_of_frames())) 

    def get_image(self):
        # Retrieve left rectified image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        self.zed.retrieve_image(self.depth, sl.VIEW.CONFIDENCE)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        # Retrieve confidence map.
        self.zed.retrieve_measure(self.confidence_map, sl.MEASURE.CONFIDENCE, sl.MEM.CPU)

        # convert zed image to numpy array
        self.img = self.image.get_data()
        self.depth_img = self.depth.get_data()
