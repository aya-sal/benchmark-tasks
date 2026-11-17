"""Prepare flat-base bottle colliders and paired scene files for yam-sim edit."""

import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
PREPPED = HERE / "assets/prepped"
# Z bounds and radii in source metres. The flat base matches the radius near
# the bottom of the measured mesh; the cap is a separate grasp surface.
CYLINDERS = {
    "lubricant": ((.0004, .205, .040), (.1995, .2466, .0367)),
    "spray": ((0.0, .151, .030), (.146, .1864, .0302)),
}


def stable_bottle(name):
    output = PREPPED / f"{name}_flat.usda"
    stage = Usd.Stage.CreateNew(str(output))
    root = stage.DefinePrim("/RootNode", "Xform")
    root.GetReferences().AddReference(f"./{name}.usda")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    material = UsdShade.Material.Define(stage, root.GetPath().AppendChild("BottlePhysics"))
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.75)
    physics.CreateDynamicFrictionAttr(.60)
    physics.CreateRestitutionAttr(0)
    for part, (low, high, radius) in zip(("BodyCollider", "CapCollider"), CYLINDERS[name], strict=True):
        shape = UsdGeom.Cylinder.Define(stage, root.GetPath().AppendChild(part))
        shape.CreateAxisAttr(UsdGeom.Tokens.z)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(high - low)
        shape.AddTranslateOp().Set(Gf.Vec3d(0, 0, (low + high) / 2))
        shape.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()


def editor_scene(path, use_flat):
    from pxr import UsdLux

    import task

    stage = Usd.Stage.CreateNew(str(path))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(850)
    table = UsdGeom.Cube.Define(stage, "/World/Table")
    table.CreateSizeAttr(1)
    table.AddTranslateOp().Set(Gf.Vec3d(.36, 0, task.TOP - .03))
    table.AddScaleOp().Set(Gf.Vec3f(1.10, 1.30, .06))
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    for name in ("box", "lubricant", "spray"):
        spec = task.BOX if name == "box" else task.SPEC[name]
        source = Path(spec.usd) if name == "box" or use_flat else PREPPED / f"{name}.usda"
        prim = stage.DefinePrim(f"/World/{name}", "Xform")
        prim.GetReferences().AddReference("./" + os.path.relpath(source, path.parent))
        pose = UsdGeom.Xformable(prim)
        pose.ClearXformOpOrder()
        if name == "box":
            xyz, yaw = spec.spawn_origin, 180
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        else:
            x, y, yaw = task.SLOTS[name]
            xyz = (x, y, task.BOX_FLOOR - spec.bmin[2] * spec.scale[2] + .001)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec.mass)
        pose.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        pose.AddRotateZOp().Set(yaw)
        pose.AddScaleOp().Set(Gf.Vec3f(*spec.scale))
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    for bottle in CYLINDERS:
        stable_bottle(bottle)
    for variant in ("original", "flat"):
        editor_scene(HERE / f"bottle-seating-{variant}.usda", use_flat=variant == "flat")
