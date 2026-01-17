import subprocess
import os
import time
import re
from PIL import Image


def run_adb_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    '''
    if result.returncode != 0:
        print(f"Command failed: {command}")
        print(result.stderr)
    '''
    return result


def get_screenshot(adb_path):
    remote_path = "/sdcard/screenshot.png"
    local_dir = "./screenshot"
    local_path = os.path.join(local_dir, "screenshot.png")
    save_path = os.path.join(local_dir, "screenshot.jpg")

    os.makedirs(local_dir, exist_ok=True)

    commands = [
        f"{adb_path} shell rm {remote_path}",
        f"{adb_path} shell screencap -p {remote_path}",
        f"{adb_path} pull {remote_path} {local_dir}"
    ]

    for cmd in commands:
        run_adb_command(cmd)
        time.sleep(0.5)

    image = Image.open(local_path)
    image.convert("RGB").save(save_path, "JPEG")
    os.remove(local_path)


def save_screenshot_to_file(adb_path, file_path="screenshot.png"):
    """
    Captures a screenshot from an Android device using ADB, saves it locally, and removes the screenshot from the device.

    Args:
        adb_path (str): Path to the adb executable.
        file_path (str): Local path to save the screenshot.

    Returns:
        str or None: Path to the saved screenshot, or None if an error occurs.
    """
    local_file = file_path
    device_file = "/sdcard/screenshot.png"

    # Ensure local directory exists
    dir_path = os.path.dirname(local_file)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    try:
        # Step 1: Remove existing screenshot on device
        run_adb_command(f"{adb_path} shell rm {device_file}")
        time.sleep(0.5)

        # Step 2: Capture screenshot
        result = run_adb_command(f"{adb_path} shell screencap -p {device_file}")
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to capture screenshot: {result.stderr.strip()}")

        # Step 3: Pull to local
        result = run_adb_command(f"{adb_path} pull {device_file} {local_file}")
        time.sleep(0.5)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull screenshot: {result.stderr.strip()}")

        # Step 4: Cleanup device file
        result = run_adb_command(f"{adb_path} shell rm {device_file}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to delete screenshot on device: {result.stderr.strip()}")

        print(f"\tAtomic Operation Screenshot saved to {local_file}")
        return local_file

    except Exception as e:
        print(str(e))
        return None


def tap(adb_path, x, y):
    run_adb_command(f"{adb_path} shell input tap {x} {y}")


def type(adb_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            run_adb_command(f"{adb_path} shell input text %s")
        elif char == '_':
            run_adb_command(f"{adb_path} shell input keyevent 66")
        elif char.isalnum():
            run_adb_command(f"{adb_path} shell input text {char}")
        elif char in '-.,!?@\'°/:;()':
            run_adb_command(f'{adb_path} shell input text "{char}"')
        else:
            run_adb_command(f'{adb_path} shell am broadcast -a ADB_INPUT_TEXT --es msg "{char}"')


def enter(adb_path):
    run_adb_command(f"{adb_path} shell input keyevent KEYCODE_ENTER")


def swipe(adb_path, x1, y1, x2, y2):
    run_adb_command(f"{adb_path} shell input swipe {x1} {y1} {x2} {y2} 500")


def back(adb_path):
    run_adb_command(f"{adb_path} shell input keyevent 4")


def home(adb_path):
    run_adb_command(f"{adb_path} shell input keyevent KEYCODE_HOME")


def switch_app(adb_path):
    run_adb_command(f"{adb_path} shell input keyevent KEYCODE_APP_SWITCH")


def close_all_apps(adb_path, wait_time=1.0):
    """
    Close all recent apps using multiple methods for maximum compatibility.
    
    Step 1: Force-stop running apps (kills processes)
    Step 2: Clear the Recent Apps screen (removes cards from task switcher)
    
    Works on all Android devices regardless of UI differences.
    """
    import time
    import subprocess
    
    # Press home first to ensure we're not in any app
    run_adb_command(f"{adb_path} shell input keyevent KEYCODE_HOME")
    time.sleep(0.3)
    
    # Step 1: Get recent tasks and force-stop each app (kills processes)
    force_stopped_count = 0
    try:
        # Get list of recent tasks/packages
        result = subprocess.run(
            f"{adb_path} shell dumpsys activity recents",
            shell=True, capture_output=True, text=True, timeout=10
        )
        
        # Extract package names from recent tasks
        packages_to_kill = set()
        for line in result.stdout.split('\n'):
            # Look for package names in the recents dump
            if 'baseIntent' in line or 'realActivity' in line or 'A=' in line:
                # Extract package name (format varies, but usually before '/')
                matches = re.findall(r'([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z][a-zA-Z0-9_]*)+)', line)
                for match in matches:
                    # Filter out system packages we shouldn't kill
                    if not any(skip in match for skip in ['launcher', 'systemui', 'inputmethod', 'keyboard', 'android.ext']):
                        packages_to_kill.add(match)
        
        # Force stop each app
        for package in packages_to_kill:
            run_adb_command(f"{adb_path} shell am force-stop {package}")
        
        force_stopped_count = len(packages_to_kill)
        
    except Exception as e:
        print(f"[CloseApps] Force-stop failed: {e}")
    
    # Step 2: Clear the Recent Apps screen (removes cards from task switcher)
    # This is necessary because force-stop doesn't remove apps from the Recent Apps view
    run_adb_command(f"{adb_path} shell input keyevent KEYCODE_APP_SWITCH")
    time.sleep(wait_time)
    
    # Swipe up multiple times to dismiss app cards from Recent Apps
    # Use coordinates suitable for most phone screens
    dismissed_count = 0
    for _ in range(15):  # Dismiss up to 15 apps
        run_adb_command(f"{adb_path} shell input swipe 540 1200 540 200 100")
        time.sleep(0.15)
        dismissed_count += 1
    
    print(f"[CloseApps] Force-stopped {force_stopped_count} apps, dismissed {dismissed_count} from Recent Apps")
    
    # Press home to ensure we're at home screen
    run_adb_command(f"{adb_path} shell input keyevent KEYCODE_HOME")
    time.sleep(0.3)


def reset_to_home(adb_path, close_apps=True, wait_time=2.0):
    """
    Reset the phone to home screen, optionally closing all apps.
    
    Args:
        adb_path: Path to ADB executable
        close_apps: If True, attempt to close all recent apps
        wait_time: Time to wait for animations
    """
    import time
    
    if close_apps:
        close_all_apps(adb_path, wait_time=1.0)
    else:
        run_adb_command(f"{adb_path} shell input keyevent KEYCODE_HOME")
    
    time.sleep(wait_time)
    print("[Reset] Phone returned to home screen")


# ========== App Detection ==========

# Mapping from Android package names to friendly app names
PACKAGE_TO_APP_NAME = {
    # Google Apps
    "com.google.android.apps.maps": "Google Maps",
    "com.google.android.gm": "Gmail",
    "com.google.android.calendar": "Google Calendar",
    "com.google.android.apps.messaging": "Messages",
    "com.google.android.dialer": "Phone",
    "com.google.android.contacts": "Contacts",
    "com.google.android.apps.photos": "Google Photos",
    "com.google.android.youtube": "YouTube",
    "com.android.chrome": "Google Chrome",
    "com.google.android.apps.docs": "Google Drive",
    "com.google.android.keep": "Google Keep",
    "com.google.android.apps.translate": "Google Translate",
    
    # System Apps
    "com.android.settings": "Settings",
    "com.android.vending": "Play Store",
    "com.android.camera": "Camera",
    "com.google.android.camera": "Camera",
    "com.android.launcher": "Home",
    "com.google.android.apps.nexuslauncher": "Home",
    
    # Common Third-Party Apps
    "com.walmart.android": "Walmart",
    "com.amazon.mShop.android.shopping": "Amazon",
    "com.ebay.mobile": "eBay",
    "com.twitter.android": "X",
    "com.instagram.android": "Instagram",
    "com.facebook.katana": "Facebook",
    "com.whatsapp": "WhatsApp",
    "com.spotify.music": "Spotify",
    "com.netflix.mediaclient": "Netflix",
    "com.ubercab": "Uber",
    "com.application.zomato": "Zomato",
    "com.yelp.android": "Yelp",
    "com.tripadvisor.tripadvisor": "TripAdvisor",
    "com.bestbuy.android": "Best Buy",
    "com.target.ui": "Target",
    "com.fandango": "Fandango",
    
    # Notes Apps
    "com.google.android.apps.docs.editors.docs": "Google Docs",
    "com.samsung.android.app.notes": "Samsung Notes",
    "com.oneplus.note": "Notes",
    "com.android.notes": "Notes",
    "com.google.android.keep": "Keep",
}


def get_current_app(adb_path) -> dict:
    """
    Get the current foreground app using ADB.
    
    Returns:
        dict with keys:
            - package: Full package name (e.g., "com.google.android.apps.maps")
            - app_name: Friendly name (e.g., "Maps")
            - activity: Current activity name
            - is_home: True if on home/launcher screen
    """
    result = {
        "package": None,
        "app_name": "Unknown",
        "activity": None,
        "is_home": False
    }
    
    try:
        # Method 1: Get resumed activity (most reliable)
        cmd_result = run_adb_command(
            f"{adb_path} shell dumpsys activity activities | grep mResumedActivity"
        )
        
        output = cmd_result.stdout.strip()
        
        if output:
            # Parse: mResumedActivity: ActivityRecord{...com.package/com.package.Activity...}
            # Extract package and activity
            match = re.search(r'(\S+)/(\S+)', output)
            if match:
                result["package"] = match.group(1)
                result["activity"] = match.group(2)
        
        # If method 1 failed, try method 2
        if not result["package"]:
            cmd_result = run_adb_command(
                f"{adb_path} shell dumpsys window | grep -E 'mCurrentFocus'"
            )
            output = cmd_result.stdout.strip()
            if output:
                match = re.search(r'(\S+)/(\S+)', output)
                if match:
                    result["package"] = match.group(1)
                    result["activity"] = match.group(2)
        
        # Map package to friendly name
        if result["package"]:
            package = result["package"]
            
            # Check direct mapping
            if package in PACKAGE_TO_APP_NAME:
                result["app_name"] = PACKAGE_TO_APP_NAME[package]
            else:
                # Try partial matching for variants
                for pkg_pattern, app_name in PACKAGE_TO_APP_NAME.items():
                    if pkg_pattern in package or package in pkg_pattern:
                        result["app_name"] = app_name
                        break
                else:
                    # Extract app name from package (last part, cleaned up)
                    parts = package.split('.')
                    if parts:
                        # Try to get a meaningful name
                        name = parts[-1].replace('android', '').replace('app', '').strip()
                        if name:
                            result["app_name"] = name.title()
            
            # Check if it's home/launcher
            if any(x in package.lower() for x in ['launcher', 'home']):
                result["is_home"] = True
                result["app_name"] = "System Operation"
    
    except Exception as e:
        print(f"[get_current_app] Error: {e}")
    
    return result


def get_current_app_name(adb_path) -> str:
    """
    Convenience function to get just the app name.
    
    Returns:
        str: Friendly app name (e.g., "Maps", "Chrome")
    """
    return get_current_app(adb_path).get("app_name", "Unknown")


def normalize_app_name(app_name: str) -> str:
    """
    Normalize app name to handle LLM variations.
    
    Examples:
        "Maps" -> "Google Maps"
        "google maps" -> "Google Maps"
        "Chrome" -> "Google Chrome"
        "Chrome Browser" -> "Google Chrome"
    """
    if not app_name:
        return "Unknown"
    
    # Convert to lowercase for matching
    app_lower = app_name.lower().strip()
    
    # Common normalizations - canonical names
    normalizations = {
        # Google Apps - use full "Google X" names
        "google maps": "Google Maps",
        "maps": "Google Maps",
        "g maps": "Google Maps",
        "google chrome": "Google Chrome",
        "chrome browser": "Google Chrome",
        "chrome": "Google Chrome",
        "google calendar": "Google Calendar",
        "calendar": "Google Calendar",
        "google photos": "Google Photos",
        "photos": "Google Photos",
        "gallery": "Google Photos",
        "google drive": "Google Drive",
        "drive": "Google Drive",
        "google keep": "Google Keep",
        "keep": "Google Keep",
        "keep notes": "Google Keep",
        "google translate": "Google Translate",
        "translate": "Google Translate",
        
        # Other Google Apps (keep original names)
        "gmail": "Gmail",
        "google mail": "Gmail",
        "youtube": "YouTube",
        "google youtube": "YouTube",
        
        # Notes apps
        "google notes": "Notes",
        "notes": "Notes",
        "samsung notes": "Notes",
        
        # Social Media
        "twitter": "X",
        "x": "X",
        "facebook": "Facebook",
        "instagram": "Instagram",
        "whatsapp": "WhatsApp",
        
        # Shopping
        "walmart": "Walmart",
        "amazon": "Amazon",
        "amazon shopping": "Amazon",
        "best buy": "Best Buy",
        "bestbuy": "Best Buy",
        "target": "Target",
        "ebay": "eBay",
        
        # System
        "settings": "Settings",
        "phone": "Phone",
        "dialer": "Phone",
        "messages": "Messages",
        "sms": "Messages",
        "camera": "Camera",
        "contacts": "Contacts",
        
        # Launcher/Home
        "home": "System Operation",
        "home screen": "System Operation",
        "launcher": "System Operation",
        "system": "System Operation",
        "system operation": "System Operation",
        "app drawer": "System Operation",
    }
    
    # Check exact match first
    if app_lower in normalizations:
        return normalizations[app_lower]
    
    # Check partial match
    for key, value in normalizations.items():
        if key in app_lower or app_lower in key:
            return value
    
    # Return title-cased version if no match
    return app_name.strip().title()
