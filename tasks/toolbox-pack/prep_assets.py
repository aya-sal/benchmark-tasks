"""Prepare the goggles without merging their temple hinges into one rigid body."""

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "assets/prepped/glasses_articulated.usda"
REFERENCE = "../ppe_eyeProtection_safetyGlasses_a01/simready_usd/sm_ppe_eyeProtection_safetyGlasses_a01_01.usd"


def prepare():
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    root = stage.DefinePrim("/RootNode", "Xform")
    root.GetReferences().AddReference(REFERENCE)
    root.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Enabled")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    geometry = root.GetChild("Geometry")
    for name in ("frame", "lens", "templeLeft", "templeRight"):
        geometry.GetChild(f"{name}_01_obj_00").SetInstanceable(False)
    frame = geometry.GetChild("frame_01_obj_00")
    UsdPhysics.ArticulationRootAPI.Apply(frame)
    for name in ("lens", "templeLeft", "templeRight"):
        joint = UsdPhysics.Joint(root.GetChild("Joints").GetChild(f"joint_{name}_01_joint_00"))
        joint.CreateBody0Rel().SetTargets([frame.GetPath()])
        joint.CreateBody1Rel().SetTargets([geometry.GetChild(f"{name}_01_obj_00").GetPath()])
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    prepare()
