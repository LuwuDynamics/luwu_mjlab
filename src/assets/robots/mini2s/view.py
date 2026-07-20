"""MuJoCo viewer for Mini2s robot."""
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

DEFAULT_XML = Path(__file__).resolve().parent / "xmls" / "scene_mini2s.xml"


def main() -> None:
    xml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML
    print(f"加载: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print(
        f"body={model.nbody} joint={model.njnt} geom={model.ngeom} actuator={model.nu}"
    )
    print("右键面板拖动滑块控制关节 | 按数字键 2、3 切换视觉/碰撞组")
    mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)


if __name__ == "__main__":
    main()
