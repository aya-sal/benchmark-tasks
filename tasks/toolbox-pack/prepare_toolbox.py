"""Build the fixed-open packing case from the stock SimReady toolbox."""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "assets/prepped/toolbox_pack_open.usda"
SOURCE = ("../storage_toolbox_standard_hardCase_d01/simready_usd/"
          "sm_storage_toolbox_standard_hardCase_d01_01.usd")
HINGE = Gf.Vec3d(-0.15407753, 0.00003459, 0.1651678)


def translate(offset):
    return Gf.Matrix4d().SetTranslate(Gf.Vec3d(*offset))


def collider(stage, root, name, low, high):
    shape = UsdGeom.Cube.Define(stage, root.GetPath().AppendChild(name))
    shape.CreateSizeAttr(1)
    shape.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(low, high)]))
    shape.AddScaleOp().Set(Gf.Vec3f(*[b - a for a, b in zip(low, high)]))
    shape.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())


def main():
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    root = stage.DefinePrim("/RootNode", "Xform")
    root.GetReferences().AddReference(SOURCE)
    root.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Enabled")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    geo = root.GetChild("Geometry")
    for part in geo.GetChildren():
        part.SetInstanceable(False)
    root.GetChild("Joints").SetActive(False)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    # The lid, handle and latch visuals rotate together about the original hinge.
    swing = translate(-HINGE) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)) * translate(HINGE)
    for name in ("top", "hand", "lock"):
        xform = UsdGeom.Xformable(geo.GetChild(f"{name}_01_obj_00"))
        transform = xform.GetLocalTransformation() * swing
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(transform)
    UsdPhysics.RigidBodyAPI.Apply(root)

    # Floor and inner wall faces from ray sections of the base mesh.
    collider(stage, root, "floor", (-.128, -.249, .000), (.129, .249, .01565))
    collider(stage, root, "front", (.112, -.249, .015), (.139, .249, .166))
    collider(stage, root, "rear", (-.151, -.249, .015), (-.112, .249, .1742))
    for name, y0, y1 in (("side_left", .229, .249), ("side_right", -.249, -.229)):
        collider(stage, root, name, (-.113, y0, .015), (.113, y1, .146))
    collider(stage, root, "open_lid", (-.213, -.264, .164), (-.134, .264, .469))
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    main()
