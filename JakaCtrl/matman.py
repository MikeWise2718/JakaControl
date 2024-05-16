import os

from pxr import Sdf, UsdShade
from typing import Tuple, List
import carb.settings


class MatMan():
    matlib = {}
    _stage = None

    def __init__(self, stage) -> None:
        self.CreateMaterials()
        self._stage = stage

    def ResetStage(self, stage):
        self._stage = stage

    def MakePreviewSurfaceTexMateral(self, matname: str, fname: str):
        # This is all materials
        matpath = "/World/Looks"
        mlname = f'{matpath}/boardMat_{fname.replace(".","_")}'
        # stage = omni.usd.get_context().get_stage()
        stage = self._stage
        material = UsdShade.Material.Define(stage, mlname)
        pbrShader = UsdShade.Shader.Define(stage, f'{mlname}/PBRShader')
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")
        stReader = UsdShade.Shader.Define(stage, f'{matpath}/stReader')
        stReader.CreateIdAttr('UsdPrimvarReader_float2')

        diffuseTextureSampler = UsdShade.Shader.Define(stage, f'{matpath}/diffuseTexture')
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        ASSETS_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
        # print(f"ASSETS_DIRECTORY {ASSETS_DIRECTORY}")
        texfile = f"{ASSETS_DIRECTORY}\\{fname}"
        # print(texfile)
        # print(os.path.exists(texfile))
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texfile)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(),
                                                                                           'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            diffuseTextureSampler.ConnectableAPI(), 'rgb')

        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        stReader.CreateInput('varname', Sdf.ValueTypeNames.Token).ConnectToSource(stInput)
        self.matlib[matname]["mat"] = material
        return material

    def SplitRgb(self, rgb: str) -> Tuple[float, float, float]:
        sar = rgb.split(",")
        r = float(sar[0])
        g = float(sar[1])
        b = float(sar[2])
        return (r, g, b)

    def MakePreviewSurfaceMaterial(self, matname: str, rgb: str):
        mtl_path = Sdf.Path(f"/World/Looks/Presurf_{matname}")

        # stage = omni.usd.get_context().get_stage()
        stage = self._stage

        mtl = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        rgbtup = self.SplitRgb(rgb)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgbtup)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        # self.matlib[matname] = {"name": matname, "typ": "mtl", "mat": mtl}
        self.matlib[matname]["mat"] = mtl
        return mtl

    refCount: int = 0
    fetchCount: int = 0
    skipCount: int = 0

    def CopyRemoteMaterial(self, matname, urlbranch, force=False):
        print(f"CopyRemoteMaterial matname:{matname} urlbranch:{urlbranch} force:{force}")
        # stage = omni.usd.get_context().get_stage()
        stage = self._stage
        baseurl = 'https://omniverse-content-production.s3.us-west-2.amazonaws.com'
        url = f'{baseurl}/Materials/{urlbranch}.mdl'
        mpath = f'/World/Looks/{matname}'
        action = ""
        # Note we should not execute the next command if the material already exists
        if force or not stage.GetPrimAtPath(mpath):
            import omni.kit.commands as okc
            okc.execute('CreateMdlMaterialPrimCommand', mtl_url=url, mtl_name=matname, mtl_path=mpath)
            action = "fetch"
            self.fetchCount += 1
        else:
            action = "skip"
            self.skipCount += 1
        mtl: UsdShade.Material = UsdShade.Material(stage.GetPrimAtPath(mpath))
        print(f"CopyRemoteMaterial {mpath} mtl:{mtl} action:{action}")
        # self.matlib[matname] = {"name": matname, "typ": "rgb", "mat": mtl}
        self.matlib[matname]["mat"] = mtl
        return mtl

    def RealizeMaterial(self, matname: str):
        try:
            typ = self.matlib[matname]["typ"]
            spec = self.matlib[matname]["spec"]
            if typ == "mtl":
                self.CopyRemoteMaterial(matname, spec)
            elif typ == "tex":
                self.MakePreviewSurfaceTexMateral(matname, spec)
            else:
                self.MakePreviewSurfaceMaterial(matname, spec)
            self.matlib[matname]["realized"] = True
        except Exception as e:
            carb.log_error(f"Exception in RealizeMaterial {matname} : {e}")

    def SetupMaterial(self, matname: str, typ: str, spec: str):
        # print(f"SetupMaterial {matname} {typ} {spec}")
        matpath = f"/World/Looks/{matname}"
        self.matlib[matname] = {"name": matname,
                                "typ": typ,
                                "mat": None,
                                "path": matpath,
                                "realized": False,
                                "spec": spec}

    def CreateMaterials(self):
        self.SetupMaterial("red", "rgb", "1,0,0")
        self.SetupMaterial("green", "rgb", "0,1,0")
        self.SetupMaterial("blue", "rgb", "0,0,1")
        self.SetupMaterial("yellow", "rgb", "1,1,0")
        self.SetupMaterial("cyan", "rgb", "0,1,1")
        self.SetupMaterial("magenta", "rgb", "1,0,1")
        self.SetupMaterial("white", "rgb", "1,1,1")
        self.SetupMaterial("black", "rgb", "0,0,0")
        self.SetupMaterial("Blue_Glass",  "mtl", "Base/Glass/Blue_Glass")
        self.SetupMaterial("Light_Blue_Glass",  "mtl", "Base/Glass/Glass_Clear_Saturated_Blue")
        self.SetupMaterial("Red_Glass", "mtl", "Base/Glass/Red_Glass")
        self.SetupMaterial("Green_Glass", "mtl", "Base/Glass/Green_Glass")
        self.SetupMaterial("Light_Green_Glass",  "mtl", "Base/Glass/Glass_Colored")
        self.SetupMaterial("Clear_Glass", "mtl", "Base/Glass/Clear_Glass")
        self.SetupMaterial("Tinted_Glass", "mtl", "Base/Glass/Tinted_Glass")
        self.SetupMaterial("Tinted_Glass_R50", "mtl", "Base/Glass/Tinted_Glass_R50")
        self.SetupMaterial("Tinted_Glass_R75", "mtl", "Base/Glass/Tinted_Glass_R75")
        self.SetupMaterial("Tinted_Glass_R85", "mtl", "Base/Glass/Tinted_Glass_R85")
        self.SetupMaterial("Tinted_Glass_R98", "mtl", "Base/Glass/Tinted_Glass_R98")

        self.SetupMaterial("Optical_Glass", "mtl", "Base/Glass/Glass_Optical")
        self.SetupMaterial("Bronze", "mtl", "Base/Metals/Bronze")
        self.SetupMaterial("Brass", "mtl", "Base/Metals/Brass")
        self.SetupMaterial("Gold", "mtl", "Base/Metals/Gold")
        self.SetupMaterial("Silver", "mtl", "Base/Metals/Silver")
        self.SetupMaterial("Iron", "mtl", "Base/Metals/Iron")
        self.SetupMaterial("Steel_Stainless", "mtl", "Base/Metals/Steel_Stainless")
        self.SetupMaterial("Steel_Blued", "mtl", "Base/Metals/Steel_Blued")
        self.SetupMaterial("Aluminum", "mtl", "Base/Metals/Aluminum")
        self.SetupMaterial("Aluminum_Brushed", "mtl", "Base/Metals/Aluminum_Brushed")
        self.SetupMaterial("Orange_Glass", "mtl", "vMaterials_2/Glass/Glass_Colored")
        self.SetupMaterial("Mirror", "mtl", "Base/Glass/Mirror")
        self.SetupMaterial("sunset_texture", "tex", "sunset.png")
        self.SetupMaterial("Andromeda", "mtl", "vMaterials_2/Paint/Carpaint/Carpaint_Shifting_Flakes")

    def GetMaterialCount(self):
        return len(self.matlib)

    def Reinitialize(self):
        for key in self.matlib:
            self.matlib[key]["realized"] = False

    def GetMaterialNames(self) -> List[str]:
        return list(self.matlib.keys())

    def GetMaterial(self, key):
        self.refCount += 1
        if key in self.matlib:
            if not self.matlib[key]["realized"]:
                self.RealizeMaterial(key)
            rv = self.matlib[key]["mat"]
        else:
            rv = None
        return rv
