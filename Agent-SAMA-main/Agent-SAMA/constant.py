# PUT YOUR API KEY
QWEN_API_KEY = "YOUR API KEY"
OPENAI_API_KEY = "YOUR API KEY"
GEMINI_API_KEY = "YOUR API KEY"
CLAUDE_API_KEY = "YOUR API KEY"
# PUT YOUR ADB PATH
ADB_PATH = "/home/ubuntu/Android/Sdk/platform-tools/adb"
# Change to "OpenAI" or "Gemini" or "Claude"
BACKBONE_TYPE = "OpenAI"


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

USAGE_TRACKING_JSONL = None


LLM_Judge = True

CAPTION_CALL_METHOD = "api"

CAPTION_MODEL = "qwen-vl-plus"

TEMP_DIR = "temp"

SCREENSHOT_DIR = "screenshot"

SLEEP_BETWEEN_STEPS = 5

MAX_RECOVERY_RETRIES = 3

DEFAULT_PERCEPTION_ARGS = {
    "device": "cuda",
    "caption_call_method": CAPTION_CALL_METHOD,
    "caption_model": CAPTION_MODEL,
    "groundingdino_model": "AI-ModelScope/GroundingDINO",
    "groundingdino_revision": "v1.0.0",
    "ocr_detection_model": "iic/cv_resnet18_ocr-detection-db-line-level_damo",
    "ocr_recognition_model": "iic/cv_convnextTiny_ocr-recognition-document_damo",
}

ATOMIC_ACTION_SIGNATURES = {
    "Open_App": {
        "arguments": ["app_name"],
        "description": lambda
            info: "If the current screen is Home or App screen, you can use this action to open the app named \"app_name\" on the visible on the current screen."
    },
    "Tap": {
        "arguments": ["x", "y"],
        "description": lambda info: "Tap the position (x, y) in current screen."
    },
    "Swipe": {
        "arguments": ["x1", "y1", "x2", "y2"],
        "description": lambda
            info: f"Swipe from position (x1, y1) to position (x2, y2). To swipe up or down to review more content, you can adjust the y-coordinate offset based on the desired scroll distance. For example, setting x1 = x2 = {int(0.5 * info.width)}, y1 = {int(0.5 * info.height)}, and y2 = {int(0.1 * info.height)} will swipe upwards to review additional content below. To swipe left or right in the App switcher screen to choose between open apps, set the x-coordinate offset to at least {int(0.5 * info.width)}."
    },
    "Type": {
        "arguments": ["text"],
        "description": lambda info: "Type the \"text\" in an input box."
    },
    "Enter": {
        "arguments": [],
        "description": lambda info: "Press the Enter key after typing (useful for searching)."
    },
    "Switch_App": {
        "arguments": [],
        "description": lambda info: "Show the App switcher for switching between opened apps."
    },
    "Back": {
        "arguments": [],
        "description": lambda info: "Return to the previous state."
    },
    "Home": {
        "arguments": [],
        "description": lambda info: "Return to home page."
    },
    "Wait": {
        "arguments": [],
        "description": lambda info: "Wait for 10 seconds to give more time for a page loading."
    }
}

INIT_SHORTCUTS = {
    "Tap_Type_and_Enter": {
        "name": "Tap_Type_and_Enter",
        "arguments": ["x", "y", "text"],
        "description": "Tap an input box at position (x, y), Type the \"text\", and then perform the Enter operation. Very useful for searching and sending messages!",
        "precondition": "There is a text input box on the screen with no previously entered content.",
        "atomic_action_sequence": [
            {"name": "Tap", "arguments_map": {"x": "x", "y": "y"}},
            {"name": "Type", "arguments_map": {"text": "text"}},
            {"name": "Enter", "arguments_map": {}}
        ]
    },
    "Tap_and_Type": {
        "name": "Tap_and_Type",
        "arguments": ["x", "y", "text"],
        "description": "Tap an input box at position (x, y) and Type the \"text\". Useful for filling forms without submitting.",
        "precondition": "There is a text input box on the screen.",
        "atomic_action_sequence": [
            {"name": "Tap", "arguments_map": {"x": "x", "y": "y"}},
            {"name": "Type", "arguments_map": {"text": "text"}}
        ]
    },
    "Long_Press": {
        "name": "Long_Press",
        "arguments": ["x", "y"],
        "description": "Long press at position (x, y) to trigger context menus or selection mode. Useful for file operations.",
        "precondition": "There is an element that supports long press actions (files, list items).",
        "atomic_action_sequence": [
            {"name": "Swipe", "arguments_map": {"x1": "x", "y1": "y", "x2": "x", "y2": "y"}}
        ]
    }
}

SHORTCUT_EXAMPLE = """
{
    "name": "Tap_Type_and_Enter",
    "arguments": ["x", "y", "text"],
    "description": "Tap an input box at position (x, y), Type the \"text\", and then perform the Enter operation (useful for searching or sending messages).",
    "precondition": "There is a text input box on the screen.",
    "atomic_action_sequence":[
        {"name": "Tap", "arguments_map": {"x":"x", "y":"y"}},
        {"name": "Type", "arguments_map": {"text":"text"}},
        {"name": "Enter", "arguments_map": {}}
    ]
}
"""

INIT_TIPS = """=== AndroidWorld Environment ===
0. Apps are NOT visible on the home screen. Use the Open_App action with app_name to launch any app directly via ADB.
1. Available apps include: contacts, simple sms messenger, markor, simple calendar pro, simple gallery pro, osmand, retro music, vlc, broccoli recipe, pro expense, clock, camera, audio recorder, simple draw pro, files, settings.
2. Do not try to find apps in an app drawer or on the home screen - always use Open_App action.
3. After Open_App, wait for the app to fully load before interacting with UI elements.

=== General Tips ===
4. Do not add any payment information. If asked to sign in, ignore it or sign in as a guest.
5. Always close pop-up windows, permission dialogs, and promotional banners when opening an app.
6. Screenshots may show partial text in text boxes from your previous input; this does not count as an error.
7. Use the Back action to dismiss dialogs, close keyboards, or navigate to previous screens.

=== Common App Patterns ===
8. Contacts: Use Open_App("contacts"). Tap "+" FAB to add new contact. Tap checkmark/Save to save.
9. Simple SMS Messenger: Use Open_App("simple sms messenger"). Tap "+" FAB to start new conversation. Enter phone number in "Add Contact or Number..." field. IMPORTANT: Tap the CHECKMARK icon (top-right, around x=1017, y=359) to CONFIRM the recipient - this opens the message compose view. Then type message in "Type a message..." field at bottom. Tap "SMS" send button (around x=1006, y=1454) to send.
10. Markor: Use Open_App("markor"). Tap "+" FAB to create note. Notes auto-save.
11. Simple Calendar Pro: Use Open_App("simple calendar pro"). Tap on date, then "+" for new event.
12. Pro Expense: Use Open_App("pro expense"). Tap "+" to add expense.
13. Broccoli Recipe: Use Open_App("broccoli recipe"). Tap "+" to add recipe.
14. Settings: Use Open_App("settings") to access system settings like WiFi, Bluetooth, Brightness.

=== Action Tips ===
15. Floating Action Buttons (FAB) are circular buttons in bottom-right for primary actions like add/create.
16. The three-dot menu (⋮) contains additional options like Delete, Share, Settings.
17. For scrollable lists, swipe up/down with sufficient distance (at least 1/3 of screen height).
18. When filling forms, complete all required fields before looking for submit/save button.
19. If the task seems complete, the system will verify automatically - don't worry about checking.
"""
