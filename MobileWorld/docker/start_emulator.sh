
# kill existing emulators
adb devices | grep emulator | cut -f1 | xargs -I {} adb -s "{}" emu kill

# Container restarts can leave PID-based AVD locks behind. A later container
# process can reuse that PID and make the emulator mistake the lock for a live
# instance. Remove only these locks, and only after no QEMU process remains in
# this container's private PID namespace.
/app/service/.venv/bin/python - <<'PY'
import os
import time
from pathlib import Path

def qemu_running():
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            if path.read_text().strip().startswith("qemu-system"):
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False

deadline = time.monotonic() + 10
while qemu_running() and time.monotonic() < deadline:
    time.sleep(0.2)
if qemu_running():
    raise SystemExit("Refusing to remove AVD locks while an emulator is still running")
avd = Path(os.environ.get("ANDROID_AVD_HOME", "/root/.android/avd")) / (os.environ["AVD_NAME"] + ".avd")
for name in ("hardware-qemu.ini.lock", "multiinstance.lock"):
    (avd / name).unlink(missing_ok=True)
PY
if [ "$?" -ne 0 ]; then
  exit 1
fi

export options="-no-audio -no-snapshot -gpu swiftshader_indirect"
# Reuse the benchmark's original state without changing virtual hardware or
# replacing the snapshot on shutdown. Keep cold boot available for other AVDs.
if [ -n "${EMULATOR_BOOT_SNAPSHOT:-}" ]; then
  snapshot_path="${ANDROID_AVD_HOME:-/root/.android/avd}/${AVD_NAME}.avd/snapshots/${EMULATOR_BOOT_SNAPSHOT}/snapshot.pb"
  if [ ! -f "$snapshot_path" ]; then
    echo "ERROR: requested boot snapshot does not exist: $snapshot_path" >&2
    exit 1
  fi
  export options="-no-audio -snapshot ${EMULATOR_BOOT_SNAPSHOT} -no-snapshot-save -gpu swiftshader_indirect"
fi
if [ "${AVD_NAME}" = "Pixel_6_API_33" ]; then
  export options="$options -grpc 8554"
fi
echo "Starting emulator with options: $options"
# Start emulator with or without VNC based on ENABLE_VNC environment variable
if [ "${ENABLE_VNC:-false}" = "true" ] || [ "${ENABLE_VNC:-false}" = "1" ]; then
    echo "VNC enabled, starting with GUI..."
    export DISPLAY=:0
    # BUGFIX: if starting vnc, set android emulator config.ini to hw.keyboard = yes. Now make it yes by default so comment out this line.
    # sed -i 's/hw.keyboard = no/hw.keyboard = yes/' /root/.android/avd/${AVD_NAME}.avd/config.ini

    nohup emulator -avd ${AVD_NAME} $options > /var/log/emulator.log 2>&1 & 
else
    echo "VNC disabled, starting headless emulator..."
    nohup emulator -avd ${AVD_NAME} -no-window $options > /var/log/emulator.log 2>&1 &
fi

# code from: https://github.com/google-research/android_world/blob/main/docker_setup/start_emu_headless.sh
BL='\033[0;34m'
G='\033[0;32m'
RED='\033[0;31m'
YE='\033[1;33m'
NC='\033[0m' # No Color
function check_emulator_status () {
  printf "${G}==> ${BL}Checking emulator booting up status 🧐${NC}\n"
  start_time=$(date +%s)
  spinner=( "⠹" "⠺" "⠼" "⠶" "⠦" "⠧" "⠇" "⠏" )
  i=0
  # Get the timeout value from the environment variable or use the default value of 600 seconds (10 minutes)
  timeout=${EMULATOR_TIMEOUT:-600}

  while true; do
    result=$(timeout 5 adb shell getprop sys.boot_completed 2>&1)

    if [ "$result" == "1" ]; then
      printf "\e[K${G}==> \u2713 Emulator is ready : '$result'           ${NC}\n"
      adb devices -l
      adb shell input keyevent 82
      return 0  # Return a 0 to indicate emulator has booted successfully
    elif [ "$result" == "" ]; then
      printf "${YE}==> Emulator is partially Booted! 😕 ${spinner[$i]} ${NC}\r"
    else
      printf "${RED}==> $result, please wait ${spinner[$i]} ${NC}\r"
      i=$(( (i+1) % 8 ))
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -gt $timeout ]; then
      printf "${RED}==> Timeout after ${timeout} seconds elapsed 🕛.. ${NC}\n"
      return 1 # Return a 1 to indicate failure if exceeded timeout
    fi
    sleep 4
  done
};

function disable_animation() {
  adb shell "settings put global window_animation_scale 0.0"
  adb shell "settings put global transition_animation_scale 0.0"
  adb shell "settings put global animator_duration_scale 0.0"
};


if check_emulator_status; then
  sleep 1
  disable_animation
  sleep 1
else
  echo "Emulator failed to start properly, exiting..."
  exit 1
fi

adb root
