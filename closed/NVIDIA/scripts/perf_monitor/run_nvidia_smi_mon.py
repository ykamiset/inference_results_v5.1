# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
import shutil
import signal
import subprocess

###########################
#   Tables
###########################


FULL_METRICS = {
    "timestamp": "The timestamp of when the query was made in format 'YYYY/MM/DD HH:MM:SS.msec'",
    "driver_version": "The version of the installed NVIDIA display driver. This is an alphanumeric string.",
    "gpu_name": " The official product name of the GPU. This is an alphanumeric string. For all products.",
    "gpu_serial": " This number matches the serial number physically printed on each board. It is a globally unique immutable alphanumeric value.",
    "gpu_uuid": " This value is the globally unique immutable alphanumeric identifier of the GPU. It does not correspond to any physical label on the board.",
    "pci.bus_id": " PCI bus id as 'domain:bus:device.function', in hex.",
    "pci.domain": " PCI domain number, in hex.",
    "pci.bus": " PCI bus number, in hex.",
    "pci.device": " PCI device number, in hex.",
    "pci.baseClass": " PCI Base Classcode, in hex.",
    "pci.subClass": " PCI Sub Classcode, in hex.",
    "pci.device_id": " PCI vendor device id, in hex",
    "pci.sub_device_id": " PCI Sub System id, in hex",
    "pstate": " The current performance state for the GPU. States range from P0 (maximum performance) to P12 (minimum performance).",
    "clocks_throttle_reasons.supported": " Bitmask of supported clock event reasons. See nvml.h for more details.",
    "clocks_throttle_reasons.active": " Bitmask of active clock event reasons. See nvml.h for more details.",
    "clocks_throttle_reasons.gpu_idle": " Nothing is running on the GPU and the clocks are dropping to Idle state. This limiter may be removed in a later release.",
    "clocks_throttle_reasons.applications_clocks_setting": " GPU clocks are limited by applications clocks setting. E.g. can be changed by nvidia-smi --applications-clocks=",
    "clocks_throttle_reasons.sw_power_cap": " SW Power Scaling algorithm is reducing the clocks below requested clocks because the GPU is consuming too much power. E.g. SW power cap limit can be changed with nvidia-smi --power-limit=",
    "clocks_throttle_reasons.hw_slowdown": " HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged.",
    "clocks_throttle_reasons.hw_thermal_slowdown": " HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of temperature being too high",
    "clocks_throttle_reasons.hw_power_brake_slowdown": " HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of External Power Brake Assertion being triggered (e.g. by the system power supply)",
    "clocks_throttle_reasons.sw_thermal_slowdown": " SW Thermal capping algorithm is reducing clocks below requested clocks because GPU temperature is higher than Max Operating Temp.",
    "clocks_throttle_reasons.sync_boost": " Sync Boost This GPU has been added to a Sync boost group with nvidia-smi or DCGM",
    "memory.total": " Total installed GPU memory.",
    "memory.reserved": " Total memory reserved by the NVIDIA driver and firmware.",
    "memory.used": " Total memory allocated by active contexts.",
    "memory.free": " Total free memory.",
    "utilization.gpu": " Percent of time over the past sample period during which one or more kernels was executing on the GPU.",
    "utilization.memory": " Percent of time over the past sample period during which global (device) memory was being read or written.",
    "utilization.encoder": " Percent of time over the past sample period during which one or more kernels was executing on the Encoder Engine.",
    "utilization.decoder": " Percent of time over the past sample period during which one or more kernels was executing on the Decoder Engine.",
    "utilization.jpeg": " Percent of time over the past sample period during which one or more kernels was executing on the Jpeg Engine.",
    "utilization.ofa": " Percent of time over the past sample period during which one or more kernels was executing on the Optical Flow Accelerator Engine.",
    "temperature.gpu": " Core GPU temperature. in degrees C.",
    "temperature.gpu.tlimit": " GPU T.Limit temperature. in degrees C.",
    "temperature.memory": " HBM memory temperature. in degrees C.",
    "power.draw": " The last measured power draw for the entire board, in watts. On Ampere or newer devices, returns average power draw over 1 sec. On older devices, returns instantaneous power draw. Only available if power management is supported. This reading is accurate to within +/- 5 watts.",
    "power.draw.average": " The last measured average power draw for the entire board, in watts. Only available if power management is supported and Ampere (except GA100) or newer devices. This reading is accurate to within +/- 5 watts.",
    "power.draw.instant": " The last measured instant power draw for the entire board, in watts. Only available if power management is supported. This reading is accurate to within +/- 5 watts.",
    "clocks.gr": " Current frequency of graphics (shader) clock.",
    "clocks.sm": " Current frequency of SM (Streaming Multiprocessor) clock.",
    "clocks.mem": " Current frequency of memory clock.",
    "clocks.video": " Current frequency of video encoder/decoder clock.",
    "clocks.applications.gr": " User specified frequency of graphics (shader) clock.",
    "clocks.applications.mem": " User specified frequency of memory clock.",
    "clocks.default_applications.gr": " Default frequency of applications graphics (shader) clock.",
    "clocks.default_applications.mem": " Default frequency of applications memory clock.",
}

DEFAULT_METRICS = [
    "timestamp",
    "pci.bus_id",
    "pstate",
    "clocks.gr",
    "clocks.mem",
    "power.draw",
    "utilization.gpu",
    "utilization.memory",
    "memory.used,temperature.gpu",
]

# 2024/03/13 23:53:32.786, 00000000:01:00.0, P0, 345 MHz, 1593 MHz, 47.34 W, 0 %, 0 %, 0 MiB, 39
pattern_removal = [" MHz,", " W,", " \%,", " MiB,"]


###########################
#   Support methods
###########################


def exec_cmd(cmd, verbose=False):
    if verbose:
        print("exec_cmd: Executing: {}".format(cmd))
    error = os.system(cmd)
    if error != 0:
        print("exec_cmd: Error running: {}".format(cmd))
    return error


def exec_cmd_bg(cmd_list, verbose=False):
    if verbose:
        print("exec_cmd_bg: executing {}", cmd_list)
    proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, shell=False)
    return proc


###########################
#   Command creation
###########################


def get_nvidia_smi_cmd_list(out_file_name, polling, metrics=None, gpu_id=None):
    # Pre-process metrics
    if metrics is None:
        # use defaut metrics
        metrics = DEFAULT_METRICS
    else:
        # add head metrics at the beggining
        head_metrics = ["timestamp", "pci.bus_id"]
        for head_metric in head_metrics:
            if head_metric in metrics:
                metrics.remove(head_metric)
        metrics = head_metrics + metrics

    # Assemble command line
    cmd_list = [
        shutil.which("nvidia-smi"),
        "--format=csv",
        "--loop-ms={}".format(polling),
        "--filename={}".format(out_file_name),
        "--query-gpu={}".format(",".join(metrics)),
    ]
    if gpu_id is not None:
        cmd_list.append(",".join(gpu_id))

    return cmd_list


###########################
#   Main method
###########################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cmd", required=True, help="Command line")
    parser.add_argument("-o", "--csv", required=True, help="Output csv file")
    parser.add_argument(
        "-g", "--gpu", nargs="+", required=False, default=None, help="Profiled GPUs"
    )
    parser.add_argument(
        "-m", "--metrics", nargs="+", required=False, default=None, help="GPU metrics"
    )
    parser.add_argument(
        "-p", "--polling", required=False, default=30, help="Polling rate (ms)"
    )
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    # Show metrics and exit if chosen
    if args.show_metrics:
        print(FULL_METRICS)
        exit(0)

    # Step 0 : launch nvidia-smi
    nvidia_smi_cmd_list = get_nvidia_smi_cmd_list(
        args.csv, args.polling, args.metrics, args.gpu
    )
    nvidia_smi_proc = exec_cmd_bg(nvidia_smi_cmd_list, args.verbose)

    # Step 1 : launch main command
    main_cmd = args.cmd
    exec_cmd(main_cmd, args.verbose)

    # Step 2: sleep and kill monitor
    if args.verbose:
        print("Sleeping for 5 sec...")
    exec_cmd("sleep 5")
    if args.verbose:
        print("Killing Pid={}".format(nvidia_smi_proc.pid))
    nvidia_smi_proc.send_signal(signal.SIGTERM)

    # Step 2: capture stdout
    out = nvidia_smi_proc.communicate()[0]
    out.decode("utf-8")

    # Step 3: post-process output
    with open(args.csv, "r") as fh:
        row_list = list()
        for line in fh:
            # remove unneeded units from columns
            for pattern in pattern_removal:
                line = re.sub(pattern, ",", line)
            row_list.append(line)
    row_list[0] = re.sub(r",\s+", ",", row_list[0])  # remove leading spaces from header
    with open(args.csv, "w") as fh:
        fh.write("".join(row_list))


if __name__ == "__main__":
    main()
