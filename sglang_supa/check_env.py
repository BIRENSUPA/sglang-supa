import os
import subprocess
from collections import defaultdict

import torch

from sglang.check_env import BaseEnv


class SUPAEnv(BaseEnv):
    """Environment checker for Biren SUPA GPU"""

    def get_info(self):
        supa_info = {"SUPA available": torch.supa.is_available()}

        if supa_info["SUPA available"]:
            supa_info.update(self.get_device_info())
            supa_info.update(self._get_supa_version_info())

        return supa_info

    def get_device_info(self):
        devices = defaultdict(list)
        for k in range(torch.supa.device_count()):
            devices[torch.supa.get_device_name(k)].append(str(k))

        gpu_info = {}
        for name, device_ids in devices.items():
            gpu_info[f"GPU {','.join(device_ids)}"] = name

        return gpu_info

    def _get_supa_version_info(self):
        from torch_supa.utils.cpp_extension import FULLSTACK_HOME

        supa_info = {"FULLSTACK_HOME": FULLSTACK_HOME}

        if FULLSTACK_HOME and os.path.isdir(FULLSTACK_HOME):
            supa_info.update(self._get_brcc_info())
            supa_info.update(self._get_supa_driver_version())

        return supa_info

    def _get_brcc_info(self):
        from torch_supa.utils.cpp_extension import FULLSTACK_HOME

        try:
            brcc = os.path.join(FULLSTACK_HOME, "bin/brcc")
            brcc_output = (
                subprocess.check_output(f'"{brcc}" --version', shell=True)
                .decode("utf-8")
                .strip()
            )
            first_line = brcc_output.splitlines()[0]
            version_str = first_line.split("(")[0].strip()
            return {"BRCC": version_str}
        except subprocess.SubprocessError:
            return {"BRCC": "Not Available"}

    def _get_supa_driver_version(self):
        try:
            output = subprocess.check_output(
                ["brsmi", "-q"],
                text=True,
            )
            for line in output.splitlines():
                if "Driver Version" in line:
                    return {"SUPA Driver Version": line.split(":", 1)[1].strip()}

            return {"SUPA Driver Version": "Not Available"}
        except subprocess.SubprocessError:
            return {"SUPA Driver Version": "Not Available"}

    def get_topology(self):
        try:
            result = subprocess.run(
                ["brsmi", "topo", "-m"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return {
                "SUPA Topology": (
                    "\n" + result.stdout if result.returncode == 0 else None
                )
            }
        except subprocess.SubprocessError:
            return {}


if __name__ == "__main__":
    env = SUPAEnv()
    env.check_env()
