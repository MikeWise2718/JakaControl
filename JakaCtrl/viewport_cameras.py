import omni
from datetime import datetime

from omni.kit.widget.viewport import ViewportWidget
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.world import World
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import capture_viewport_to_file

# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

class ViewPortCameras():

    def __init__(self, uibuilder=None):
        super().__init__()
        self.cam_viewport_1 = None
        self.cam_viewport_2 = None

    def make_cage_cam_views(self,cagecamlist, wintitle, wid, heit):

        nrobcam = len(cagecamlist)
        camviews = omni.ui.Window(wintitle, width=wid, height=heit+20) # Add 20 for the title-bar

        with camviews.frame:
            if nrobcam==0:
                omni.ui.Label("No Robot Cameras Found (camlst is empty)")
            else:
                with omni.ui.VStack():
                    vh = heit / nrobcam
                    cnt=0
                    for camname in cagecamlist:
                        cam = cagecamlist[camname]
                        viewport_widget = ViewportWidget(resolution = (wid, vh))

                        if cnt==0:
                            # Control of the ViewportTexture happens through the object held in the viewport_api property
                            self.cam_viewport_1 = viewport_widget.viewport_api

                            # We can reduce the resolution of the render easily
                            self.cam_viewport_1.resolution = (wid, vh)

                            # We can also switch to a different camera if we know the path to one that exists
                            self.cam_viewport_1.camera_path = cam["usdpath"]
                            cnt+=1
                        else:
                             # Control of the ViewportTexture happens through the object held in the viewport_api property
                            self.cam_viewport_2 = viewport_widget.viewport_api

                            # We can reduce the resolution of the render easily
                            self.cam_viewport_2.resolution = (wid, vh)

                            # We can also switch to a different camera if we know the path to one that exists
                            self.cam_viewport_2.camera_path = cam["usdpath"]

                            cnt=0
        return camviews
    
    def take_snapshot(self):
        now = datetime.now()
        string = now.strftime('%Y-%m-%d--%H-%M-%S')     
        filepath = r"C:\temp\cam1_" + string + r".png"
        self.take_snapshot_cam_1(filepath)
        filepath = r"C:\temp\cam2_" + string + r".png"
        self.take_snapshot_cam_2(filepath)

    def take_snapshot_cam_1(self,filepath):
        try:
            capture_viewport_to_file(self.cam_viewport_1, filepath)  
        except:
            print("error saving take_snapshot_cam_1 snapshot")
        
    def take_snapshot_cam_2(self,filepath):
        try:
            capture_viewport_to_file(self.cam_viewport_2, filepath)
        except:
            print("error saving take_snapshot_cam_2 snapshot")

