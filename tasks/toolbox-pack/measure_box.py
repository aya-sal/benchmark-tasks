"""Print the D00443 collider bounds and safety-glasses joints with OpenUSD."""

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

ASSETS = Path(__file__).resolve().parent / "assets"


def main():
    box = Usd.Stage.Open(str(ASSETS / "D00443/simready_usd/D00443.usd"))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    print("D00443 collider bounds in asset coordinates (m)")
    for prim in Usd.PrimRange.Stage(box, Usd.TraverseInstanceProxies()):
        if "/Colliders_00/" not in str(prim.GetPath()) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        limits = [tuple(round(float(v), 5) for v in point) for point in (extent.GetMin(), extent.GetMax())]
        divider = "divider" if prim.GetName() in {"Collider_3", "Collider_12", "Collider_15", "Collider_17"} else ""
        print(prim.GetName(), *limits, divider)

    source = next(ASSETS.glob("ppe_eyeProtection*/simready_usd/sm*.usd"))
    print("\nSafety glasses joints")
    for file in (source, ASSETS / "prepped/glasses_articulated.usda"):
        stage = Usd.Stage.Open(str(file))
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        print(file.name, "articulation roots", roots)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint = UsdPhysics.RevoluteJoint(prim)
                print(" ", prim.GetName(), joint.GetAxisAttr().Get(), joint.GetLocalPos0Attr().Get(),
                      joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
            elif prim.IsA(UsdPhysics.FixedJoint):
                print(" ", prim.GetName(), "fixed")


if __name__ == "__main__":
    main()
