"""Repair well-specified action conditions and retain ambiguous cases for review.

Rules use action intent and task context, never the after image, verdict, or
failure reason to decide the expected outcome. These are rule-based revisions,
not a claim of exhaustive visual annotation. Original datasets stay untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


MANUAL = {
    ("train", 1): (
        "Gmail's compose screen must provide an editable To recipient field that can be focused and accept the intended email address.",
        "The To field contains linqg327@gmail.com as the confirmed recipient. Opening a sender-account menu or showing this address only in the From field does not satisfy the step.",
        "visually_reviewed_recipient_entry",
    ),
    ("train", 25): (
        "The app drawer is open and its app grid can scroll upward to show later entries.",
        "The app grid scrolls and exposes additional app icons for the continued search for Google Play Store. Finding or launching Play Store is not required in this single swipe.",
        "visually_reviewed_drawer_scroll",
    ),
    ("train", 92): (
        "The Directions control associated specifically with Pâtisserie Coucou Gâteau must be visible and selectable in the bakery results.",
        "Google Maps displays route options whose destination is Pâtisserie Coucou Gâteau. A route to a different bakery does not satisfy this step.",
        "text_reviewed_named_destination",
    ),
    ("train", 97): (
        "Booking.com's departure-location selector is visible and can be opened for editing.",
        "The departure-location editor opens with an editable airport/city field. Entering and choosing Chicago is a subsequent action.",
        "text_reviewed_departure_editor",
    ),
    ("train", 102): (
        "The Processor control in the ASUS Zenbook A14 product specifications must be visible and selectable.",
        "The processor specifications are displayed or expanded for inspection. This tap does not itself establish whether the laptop has an Nvidia GPU.",
        "text_reviewed_specification_tab",
    ),
    ("train", 155): (
        "Android's app-switching gesture must be available and the Clock app must be accessible through app switching.",
        "The Clock app or its selectable recent-app card is brought into view. Setting or starting the timer is a separate step.",
        "text_reviewed_app_switching_step",
    ),
    ("test", 40): (
        "The task.html file is visible and selectable in the Downloads folder.",
        "Opening task.html presents the Open with chooser so Chrome can be selected next, or opens the file directly in Chrome if that association already exists.",
        "visually_reviewed_open_with_intermediate_step",
    ),
    ("test", 445): (
        "The Markor file-selection or search interface has focus on 2023_08_21_friendly_koala.md and accepts Enter to open it.",
        "The requested file opens in the Markor editor, ready for subsequent text selection and copying.",
        "text_reviewed_enter_to_open_file",
    ),
    ("test", 821): (
        "The Markor search field has focus and contains the query proud_cat_edited.txt.",
        "Enter submits that filename query and displays matching results or opens the matching file. Later copying or merging is a separate operation.",
        "text_reviewed_enter_to_search",
    ),
}
# Text-reviewed exceptions whose immediate operation is explicit but whose old
# conditions overreach. These are not represented as screenshot-reviewed rows.
MANUAL_TEXT = [
    ("train", [82], "The Keep Notes editor is open with its back control available.", "The current note is retained and the notes list is displayed. Creating or editing another note is not required."),
    ("train", [135], "The selected hotel's Map control is available.", "The selected hotel's location is shown on the map for inspection; its distance to Green Street need not already have been verified."),
    ("train", [143, 278], "The meeting's required details are filled and the Save or Done confirmation control is enabled.", "The meeting is saved and its confirmation or sharing information becomes available. Copying or sending the invitation requires later actions."),
    ("train", [145], "The chosen hotel's room options are visible and a room-selection confirmation is available.", "The intended room selection is accepted and the next booking step appears. Completing payment or the entire booking is not required."),
    ("train", [176], "The RAM Size filter control is visible in the laptop search interface.", "The RAM-size choices open for selection. Choosing a threshold and finding a qualifying laptop are subsequent actions."),
    ("train", [177], "The intended image is available as the first thumbnail in Today within the Pixel 7 Pro folder.", "That image is selected or opened in a preview for the wallpaper workflow. Applying it as wallpaper is a separate action."),
    ("train", [237], "The reminder editor contains the requested Winter Olympics Opening Ceremony details for February 6, 2026, and Save is available.", "The configured reminder is saved and is visible in the calendar or its saved-event confirmation."),
    ("train", [477], "The Create Passcode button is available in the X Chat setup interface.", "The passcode-entry interface opens. Entering and confirming the four digits can require later actions."),
    ("train", [495], "The catan board game suggestion is visible and selectable in the Temu search interface.", "The catan board game suggestion is applied and its product search begins or results appear. Comparing prices is a later step."),
    ("train", [506], "The Set up X Chat button is visible and available in X.", "The X Chat setup flow opens at its next configuration step. Completing setup or sharing a post can require further actions."),
    ("train", [556], "The intended Bose QC45 product listing is visible and can be opened.", "That product's detail page opens. Locating and reading its detailed reviews can require later actions."),
    ("train", [571], "Zoom's launch warning is visible with Continue to use available.", "The warning is dismissed and Zoom's interface becomes accessible. Scheduling the meeting is a later step."),
    ("train", [616], "The Hyatt Place Champaign/Urbana detail page offers Select rooms.", "The hotel's room options appear for inspection and selection. Final booking confirmation is not required by this tap."),
    ("train", [633], "The Black Myth Wukong @BlackMythWukBSC result is visible and selectable.", "The selected result opens to expose its content for inspection. Collecting and summarizing highlights remain subsequent work."),
    ("train", [684], "The privacy prompt provides a Customize control.", "The privacy-customization options open for inspection or adjustment. Reaching the app's main screen can require additional choices."),
    ("train", [762, 782], "The browser search field is editable and can receive focus.", "The query Palo Alto, CA museums is entered in the intended search field and submitted to initiate its search. Selecting attractions or completing the itinerary requires later actions."),
    ("train", [829], "Chrome's browser search field is available and editable for the intended query.", "The query next Winter Olympics opening ceremony date is entered and submitted in the intended browser search interface. Entering text in an unrelated app does not satisfy the step."),
    ("val", [50], "Zoom's Schedule control is available.", "The meeting-scheduling form opens. Filling its details and saving the meeting are later steps."),
    ("val", [71], "The Circe by Madeline Miller search is active and Save this search is available.", "The search-saving flow opens or confirms that this query is saved. A sign-in requirement is reported if it blocks the operation; the book itself is not added to a watchlist by this action."),
    ("test", [181], "The requested image is open with the Lens control available.", "Lens opens for inspecting the image and selecting recognized text. Copying text or creating a note requires later actions."),
    ("test", [192, 647], "The Downloads folder displays task.html as an available file that can be opened.", "Opening task.html presents the app chooser so its handler can be selected, or opens the file directly in Chrome if that association exists. The webpage's later interactions are separate steps."),
    ("test", [198, 825], "Chrome's welcome/setup screen offers Accept & continue.", "The current setup consent is accepted and Chrome advances to its next setup screen or browser interface. Navigating the webpage's grid is a later action."),
    ("test", [199], "The deletion confirmation targets the Rental Income expense and CONFIRM is enabled.", "The Rental Income expense is deleted, with a confirmation or updated expense list reflecting its removal."),
    ("test", [404], "Chrome's welcome screen offers Use without an account.", "Chrome proceeds without signing in, advancing to the browser or next setup step. The webpage may still require another action to open."),
    ("test", [410], "The second Chicken Alfredo Pasta entry is visible and selectable.", "That duplicate entry becomes selected or opens for inspection and management. Deleting it is a later action unless the interface explicitly confirms deletion."),
    ("test", [450], "Markor's plus control is visible and enabled.", "The creation menu or a new-item form opens. Creating and naming a specific file or folder can require further actions."),
    ("test", [489], "The expense-list filter control is visible and enabled.", "The filter options open for inspection. Bulk selection is not required to become enabled by opening this menu."),
    ("test", [503], "The move destination selector lists MeetingMinutes and the intended note has been chosen for moving.", "MeetingMinutes becomes the chosen destination or the move is confirmed there. Selecting the destination alone is valid progress if a separate confirmation remains."),
    ("test", [507], "The pending move targets shy_king_copy.md and the MeetingMinutes destination, with its confirmation checkmark available.", "The move is confirmed and shy_king_copy.md is transferred to MeetingMinutes."),
    ("test", [557], "The Audio Recorder onboarding screen has Get started available.", "The app advances to its next setup/permission step or recording interface. Recording and saving audio are later actions."),
    ("test", [575], "The Open with chooser for task.html lists Chrome as an available handler.", "Chrome becomes the selected handler or opens task.html. A separate Just once or Always confirmation may still be required."),
    ("test", [587], "Audio recording is active and Stop is available.", "The recording stops and the recorded clip or its save interface becomes available. Naming or saving the clip can be a later step."),
    ("test", [636, 1080], "The intended document's text-selection controls are visible and Select all is available.", "All text in the intended document becomes selected. Copying that selection requires another action."),
    ("test", [686], "The timer keypad is active and the 2 key is available.", "One 2 keypress updates the timer entry according to its current keypad state. The entire 16-minute duration need not be entered by this single keypress."),
    ("test", [702], "The recipe image thumbnail is visible in the Internal folder and can be opened.", "The selected recipe image opens for viewing. Activating OCR and copying recognized text are later steps."),
    ("test", [726], "Chrome's Refresh control is available for the current webpage.", "The webpage reloads or displays loading progress. A broken script or missing control is not guaranteed to be repaired merely by refreshing."),
    ("test", [769], "OsmAnd is available as a selectable entry in the app switcher.", "OsmAnd is brought to the foreground. Adding the requested favorite location is later work."),
    ("test", [800], "The Rental Income expense is visible in the Recent section and can be selected.", "Rental Income becomes selected or its details open for the next management step. This tap does not itself have to delete the expense."),
    ("test", [803], "The Stopwatch tab is visible with its start/play control enabled.", "The stopwatch starts running and its elapsed time begins advancing."),
    ("test", [876], "The requested webpage is open with loading or rendering still pending.", "The webpage finishes loading or displays further loading/rendering progress. Waiting does not itself activate its button or guarantee that a broken script is repaired."),
    ("test", [931, 933, 1006], "The recipe Title field is focused and the clipboard contains the Recipe: Thai Peanut Noodle Salad entry to be pasted.", "That clipboard entry is inserted into the Title field. Saving the recipe and completing its other fields are later operations."),
    ("test", [992], "The image-handler chooser lists Photos as an available app for the requested image.", "Photos becomes the selected handler or opens the image. Choosing OCR or extracting text is a later step."),
    ("test", [1053], "The recipe-list menu offers All recipes.", "The All recipes category is selected and its recipe list is displayed. Selecting this category alone does not guarantee alphabetical sorting or duplicate deletion."),
]
for _split, _lines, _pre, _post in MANUAL_TEXT:
    for _line in _lines:
        MANUAL[(_split, _line)] = (_pre, _post, "text_reviewed_specific_operation")
MAIL_REASON = (
    "The action opened the sender-account dropdown, showing linqg327@gmail.com "
    "and spearconcordia@gmail.com as account options. The To field remains empty; "
    "the intended recipient was not entered or confirmed."
)
KNOWN_ISSUES = {
    ("train", 2): "The supplied subgoal says to return home, while the action and Yes reason describe searching for attractions. Conditions can follow this immediate search action, but the subgoal/label alignment needs separate review.",
    ("train", 138): "The alleged Sort control is described as a sun-shaped icon near the status bar; the before-state text also says results are still loading. Verify the tapped control and Yes label against screenshots.",
    ("train", 155): "The action describes switching apps and setting a three-hour timer, while the old annotation refers to three minutes. The revised conditions cover only switching apps; timing intent needs separate review.",
    ("train", 743): "The action names Location settings while the goal requires Network & Internet/Data Saver. The intended control/goal relation is inconsistent and was not guessed.",
    ("test", 686): "The description says key 2 starts a 16-minute timer. The one-key condition is repaired, but the keypad-state/goal alignment still needs image review.",
    ("test", 1053): "Selecting All recipes does not establish alphabetical sorting. The local navigation condition is repaired; the original label/reason should be reviewed if they depend on sorting.",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_of(row):
    text = row["messages"][0]["content"][0]["text"]
    return json.loads(text.split("=== Context ===\n", 1)[1].split("\n\n=== Image Reference ===", 1)[0])


def first_clause(description):
    return re.split(r'(?<=[.!?"`])\s+(?=This\b|After\b|Once\b|If\b|The goal\b|I will\b|I can\b|Repeated\b)|,? (?:allowing|enabling)\b| so that| and (?:proceed|prepare|continue)\b', description, maxsplit=1)[0].strip().rstrip(".")


def local_effect(effect):
    effect = first_clause(effect)
    if re.match(r"open|display|reveal", effect, re.I) and re.search(r"dialog|picker|menu|form|editor|search interface|details page", effect, re.I):
        effect = re.split(r" and (?:adjust|enter|select|set|add|create|save|copy|paste|fill|access|read|review)\b", effect, maxsplit=1)[0]
    return effect


def named_field(description):
    match = re.search(r'["`]([^"`\n]{1,70})["`]\s+(?:input\s+)?(?:field|box)', description)
    if match:
        return json.dumps(match.group(1), ensure_ascii=False) + " field"
    if re.search(r"search (?:bar|field|box)", description, re.I):
        return "search field"
    if re.search(r"address (?:bar|field)", description, re.I):
        return "browser address field"
    return "intended input field"


def intended_app(action, corrupted):
    args = action.get("arguments") or {}
    if not corrupted and isinstance(args.get("app_name"), str):
        return args["app_name"].strip()
    description = action["description"]
    match = re.search(r'(?:open|launch|switch(?:ing)? (?:back )?to|navigate (?:back )?to|return to) (?:the )?["`]?(.+?)["`]?(?: app\b| browser\b)', description, re.I)
    if match and len(match.group(1)) < 55:
        return match.group(1).strip('"` *')
    return None


def typed_text(action, corrupted, context=None):
    if not corrupted:
        value = (action.get("arguments") or {}).get("text")
        return value if isinstance(value, str) and value else None
    # Never promote a deliberately corrupted argument to the expected value.
    match = re.search(r'(?:typ(?:e|ing)|enter|input)\s+(?:the (?:word|text|query)\s+)?["`]([^"`]+)["`]', action["description"], re.I)
    if not match:
        return None
    value = match.group(1)
    # Some prose places the sentence's comma inside the quoted search query.
    # Remove it only when independently supported by the supplied task text.
    if value.endswith(","):
        intended = value[:-1]
        task_text = " ".join((context or {}).get(k, "") for k in ("goal", "subgoal"))
        if intended.casefold() in task_text.casefold():
            value = intended
        else:
            return None
    return value


def explicit_tap_effect(description):
    allowed = r"(?:open|display|activate|close|dismiss|select|highlight|show|reveal|navigate|return|bring|focus|insert|paste|delete|remove|save|add|start|pause|stop|resume|enable|disable|expand|collapse|update|set|submit|load|clear|mark|toggle|copy|increment|change|apply|confirm|view|access|send|launch|play|execute|initiate|refresh|share|publish|accept|bypass|center|switch|position|capture|exit|include|skip|deny)\b"
    patterns = [r"(?:This action|This) (?:is expected to|should|will) (.+)", r"I expect (?:this (?:action|tap) to )(.+)"]
    for sentence in re.split(r"(?<=\.)\s+(?=[A-Z])", description):
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                effect = local_effect(match.group(1))
                if re.match(allowed, effect, re.I) and not re.search(r"complete (?:the|this) task|finalize the task|five times|all (?:five|three|recipes|expenses)", effect, re.I):
                    return effect
    # Match an immediate purpose outside quoted UI names (e.g. "Add to a track").
    first = first_clause(description)
    masked = re.sub(r'"[^"\n]*"|`[^`\n]*`', lambda m: "_" * len(m.group()), first)
    masked = re.sub(r"\b(?:next|adjacent) to\b", lambda m: "_" * len(m.group()), masked, flags=re.I)
    # Only the first purpose clause is eligible. In "to check if it provides
    # options to enable X", enabling X is a speculative later operation.
    match = re.search(r"\bto\s+", masked, re.I)
    if match:
        start = match.end()
        attempt = re.match(r"attempt to\s+", masked[start:], re.I)
        if attempt:
            start += attempt.end()
    if match and re.match(allowed, masked[start:], re.I):
        effect = first[start:]
        effect = re.split(r",? (?:allowing|enabling|where|which)| and (?:check|read|review|note|save|proceed|continue)\b", effect, maxsplit=1)[0]
        if not re.search(r"complete (?:the|this) task|finalize|five times|all (?:five|three|recipes|expenses)", effect, re.I):
            return local_effect(effect)
    return None


def tap_target(description):
    first = first_clause(description)
    masked = re.sub(r'"[^"\n]*"|`[^`\n]*`', lambda m: "_" * len(m.group()), first)
    masked = re.sub(r"\b(?:next|adjacent) to\b", lambda m: "_" * len(m.group()), masked, flags=re.I)
    match = re.search(r"(?:tap|click|press)(?:ping)?\s+(?:on\s+)?(.+?)(?=\bto\s+|\bat coordinates|\bat position|\blocated at|\blocated in|\blocated on)", masked, re.I)
    if match and len(match.group(1)) < 180:
        return first[match.start(1):match.end(1)].strip()
    match = re.search(r'["`]([^"`]{1,70})["`]\s+(button|icon|tab|option|field|entry)', description)
    if match:
        return json.dumps(match.group(1), ensure_ascii=False) + " " + match.group(2)
    return None


def propose(context, corrupted=False):
    """Return a supported condition pair or None; no label/outcome argument."""
    action = context["action"]
    kind, description = action["type"], action["description"]
    lower = description.lower()
    first = first_clause(description).lower()
    if description in ("", "None"):
        return None
    # Corruptions may replace the executable type while leaving intended action
    # text intact. Use explicit intent to set expectations, preserving the input.
    if corrupted and kind == "Swipe" and re.search(r"\btap|\btapping", first) and not re.search(r"swip|scroll|drawing", first):
        kind = "Tap"
    elif corrupted and kind == "Tap" and re.search(r"\bswipe|\bscroll", first):
        kind = "Swipe"
    if kind == "Home" and "home" in first:
        return ("Android Home navigation is available from the current foreground interface.",
                "The Android launcher becomes visible, allowing the next app to be located. No app launch or later task operation is required by this Home action.", "home_navigation")
    if kind == "Unknown" and isinstance((action.get("arguments") or {}).get("raw"), str):
        raw = action["arguments"]["raw"]
        try:
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            embedded = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if embedded.get("name") == "Open_App_and_Wait":
            app = embedded.get("arguments", {}).get("app_name")
            if isinstance(app, str) and app:
                return (f"{app} is installed and the app-opening operation is supported by the execution interface.",
                        f"{app} opens and its initial interface finishes loading. Later in-app work is not part of this opening-and-waiting step.", "declared_open_app_and_wait")
    if kind in ("Open_App", "Switch_App"):
        app = intended_app(action, corrupted)
        if kind == "Switch_App":
            return ("Android's recent-apps or app-switching interface is available for the intended application.",
                    "The app switcher opens for locating the intended app, its selectable card is exposed, or that app becomes foreground. Later work inside the app is a separate step.", "switch_app_navigation")
        if app:
            return (f"{app} is installed and can be launched through Android's app-opening interface.",
                    f"{app} is brought to the foreground with its interface or initial app dialog visible. Subsequent searches, edits, or other in-app work are separate actions.", "app_launch_intent")
    if kind in ("Switch_To_Notes_And_Create_Text", "Create_New_Text_Note") and "keep notes" in lower:
        return ("Keep Notes is available and the declared shortcut can open its new-text-note editor.",
                "A new text-note editor is open in Keep Notes and ready for entry. Typing the task's findings and finishing the note are later actions.", "declared_new_note_shortcut")
    if kind == "Locate_And_Tap_App_In_Folder":
        match = re.search(r"tap the (.+?) app icon", description, re.I)
        if match:
            app = match.group(1).strip('"`')
            return (f"The folder containing {app} is accessible and its app icon can be selected after the folder is opened.",
                    f"The folder exposes the {app} icon or {app} opens. Later activity within the app is not required for this app-access step.", "folder_app_access")
    if kind == "Wait":
        if re.search(r"download|install", lower):
            return ("The requested app's download or installation is already in progress and can continue while the agent waits.",
                    "The pending download or installation advances or completes, with its status available for inspection. Waiting does not itself launch or configure the app.", "wait_for_install_progress")
        if re.search(r"remain|stable|stop at|task (?:is|has been)|task.*completed|screen.*displayed|screen already shows", lower):
            return ("The requested final screen or content is already displayed, so preserving it is appropriate for the task.",
                    "That task-satisfying screen remains available to the user during the wait. Waiting alone cannot satisfy a result that was missing beforehand.", "wait_preserves_final_screen")
        if "review" in lower:
            return ("The content relevant to the requested review is visible and legible.",
                    "The relevant content remains available for inspection during the pause. The wait does not by itself log, edit, or save the reviewed information.", "wait_for_reading")
    if kind == "Back":
        if "keyboard" in first:
            return ("The on-screen keyboard is visible and Android Back can dismiss it.",
                    "The keyboard is dismissed so the underlying interface can be used.", "back_dismiss_keyboard")
        return ("Back navigation or dismissal is available in the current screen or overlay.",
                "The current overlay closes or navigation returns one level to the preceding interface. Completing the later subgoal is not required by this Back action.", "back_navigation")
    if kind in ("Tap_Type_and_Enter", "Tap_and_Type", "Type"):
        value = typed_text(action, corrupted, context)
        if value is None:
            return None
        field = named_field(description)
        pre = f"The {field} is editable and can receive focus for the intended text entry."
        post = f"The {field} contains the intended text {json.dumps(value, ensure_ascii=False)}, with unrelated text not substituted for it."
        if kind == "Tap_Type_and_Enter":
            post += " The input is confirmed; for a search field, the submitted query initiates its search."
        return pre, post, "intended_text_entry"
    if kind == "Swipe":
        if "drawing" in lower or "draw a line" in lower:
            match = re.search(r"using the (.+?) color", description, re.I)
            color = match.group(1) if match else "selected"
            return (f"The drawing canvas is active and the {color} drawing color is selected.",
                    f"A {color} stroke appears along the gesture's path on the canvas. Other strokes and submission are later actions.", "single_drawing_stroke")
        if re.search(r"(?:open|access|reveal) the app drawer", lower):
            return ("The launcher is visible and accepts the gesture for opening the app drawer.",
                    "The app drawer opens and exposes its app grid. Locating or launching the target app can take further actions.", "open_app_drawer")
        if "app switcher" in lower:
            return ("The recent-apps interface or its system switching gesture is available and contains the intended app.",
                    "The recent-apps view advances toward the intended app, exposing its selectable card or bringing that app to the foreground. Later in-app work is not required by this navigation gesture.", "navigate_recent_apps")
        if "dismiss" in lower:
            return ("The current prompt supports the intended dismissal gesture or contains additional relevant controls that the gesture can expose.",
                    "The targeted prompt closes or an actionable dismissal control is revealed. Merely changing unrelated content is not sufficient.", "gesture_dismissal_progress")
        if "navigate back" in lower or "return to" in lower and "close" in lower:
            return ("The current view supports the intended back or dismissal gesture.",
                    "The current view is dismissed and the preceding interface becomes accessible. Later searching or editing remains separate.", "gesture_back_navigation")
        if "app drawer" in lower or re.search(r"(?:more|additional) apps", lower):
            return ("The launcher or app drawer is visible and can reveal additional app icons through the intended gesture.",
                    "The visible app grid or launcher page changes to expose additional icons for inspection. The target app need not already be located or launched.", "browse_app_icons")
        if "home screen" in lower:
            return ("A launcher page is visible and an adjacent launcher page can be reached by the gesture.",
                    "An adjacent launcher page becomes visible so more app icons can be inspected. Opening an app is a subsequent step.", "launcher_page")
        if re.search(r"scroll|reveal|more content|more results|more posts|more details|search results|more product listings", lower):
            return ("The intended content area is scrollable and contains further content in the direction requested by the action.",
                    "The visible content advances in the intended direction, exposing additional material relevant to the ongoing search or inspection. One scroll need not find the final item or finish reading or saving it.", "scroll_intermediate_progress")
    if kind == "Long_Press":
        quoted = re.search(r'["`]([^"`]{1,100})["`]', description)
        target = json.dumps(quoted.group(1), ensure_ascii=False) if quoted else "the intended text or item"
        return (f"The long-press target ({target}) is visible and supports selection or contextual actions.",
                f"Selection highlighting, selection handles, or a context menu appears for {target}. Copying, deleting, moving, or replacing the content can require subsequent actions.", "long_press_selection_step")
    if kind not in ("Tap", "Dismiss_Popup"):
        return None
    field = named_field(description)
    app_icon = re.search(r'(?:tap|tapping) (?:on )?(?:the )?["`]?(.+?)["`]? (?:app )?icon', description, re.I)
    if app_icon and re.search(r"open(?:ing)? (?:the )?(?:app|it|browser)|launch(?:ing)? the app|app (?:will open|launches)|reopen|switch to it", lower):
        app = app_icon.group(1).strip('"` *')
        if len(app) < 60 and not re.search(r"search|three|menu|clipboard|back|arrow|text associated", app, re.I):
            return (f"The launcher entry for {app}, or a folder containing it, is accessible for the intended tap.",
                    f"{app} opens, or its enclosing folder expands to expose the {app} icon for the next tap. Opening a different app does not satisfy this step.", "app_icon_or_enclosing_folder")
    if kind == "Dismiss_Popup" or (re.search(r"dismiss|close", first) and re.search(r"pop-up|popup|prompt|overlay|dialog", first)):
        target = tap_target(description) or "the indicated dismissal control"
        return (f"The relevant prompt or overlay is visible and {target} can be activated.",
                "The targeted prompt or overlay is dismissed and the underlying interface becomes available. Subsequent task operations are separate steps.", "dismiss_overlay_only")
    if "clipboard icon" in first and not re.search(r"\bcopy\b", first):
        return (f"The {field} is being edited and its keyboard clipboard control is accessible.",
                "The clipboard panel opens and exposes entries for selection, or the intended clipboard entry is inserted directly. Opening the panel is useful progress even when another tap is needed to paste.", "open_clipboard_panel")
    if "backspace" in lower:
        if "selected text" in first:
            post = f"The selected text in the {field} is removed. Pasting or entering replacement text is a later action."
        else:
            post = f"One character before the cursor, or the current selected range, is deleted from the {field}. A single backspace need not clear the entire field or enter replacement text."
        return (f"The {field} has input focus with deletable text, and its backspace key is available.", post, "single_backspace")
    if re.search(r'"Click Me"|\bclick me\b', description, re.I):
        return ("The Click Me button is visible and enabled on the webpage.",
                "One button activation produces the next displayed number. Collecting five numbers or performing later calculations is not required by this single tap.", "single_number_button_click")
    if "paste" in first and not re.search(r"open.*clipboard|clipboard panel|clipboard menu", first):
        return (f"The {field} has focus, the intended clipboard content is available, and its paste control is accessible.",
                f"The intended clipboard content is inserted into the {field}. Saving, renaming, or submitting the document remains a separate operation.", "paste_current_content")
    if (re.search(r"activat|active|focus|enable editing|bring up the keyboard|cursor|allow input|enabling me to type", lower)
            and re.search(r"field|search bar|input box|address bar", first)
            and not re.search(r"(?:and|,) (?:type|enter|paste)\b", first)):
        return (f"The {field} is visible and can be selected for editing.",
                f"The {field} receives input focus or its editor opens, with text entry available. This tap need not enter, replace, or submit text.", "focus_field_only")
    if re.search(r"magnifying glass|search icon|\"search\" icon", first) and re.search(r"open|access|activat|enabling the search", lower):
        return ("The search control is visible and enabled in the current interface.",
                "The search interface opens or its input gains focus. Entering a query and selecting a result are later steps.", "open_search_interface")
    if re.search(r"hamburger menu|three[- ]dot|three horizontal|three-line|menu button|menu icon", first):
        return ("The indicated navigation or overflow-menu control is visible and enabled.",
                "The corresponding menu opens and exposes its options for inspection. A desired option need not already be selected or executed.", "open_menu_only")
    if re.search(r"filter/sort icon|sorting icon|filter icon|sort icon", first):
        return ("The indicated sorting or filtering control is visible and enabled.",
                "The sort/filter options open for inspection. Opening this interface need not enable bulk selection, change a filter, or delete any item.", "open_sort_filter_options")
    key_match = re.search(r'["`]([0-9A-Za-z.,]|00)(?:["`])\s+(?:key|button)', description)
    if key_match and re.search(r"keypad|keyboard|digit|passcode", lower):
        key = key_match.group(1)
        return (f"The intended numeric/text entry has focus and the {key} key is available.",
                f"One {key} key activation updates the focused entry at its current cursor or selection position. The full number, timer value, or passcode need not be completed by this one keypress.", "single_keypress")
    folder = re.search(r'["`]([^"`]{1,100})["`]\s+(?:directory|folder)', description)
    if folder and re.search(r"navigate into|open it|navigate to|access|contents", lower) and not re.search(r"create|new folder|delete|move", first):
        name = json.dumps(folder.group(1), ensure_ascii=False)
        return (f"The folder entry {name} is visible and can be opened.",
                f"The contents of {name} are displayed. Finding, counting, or editing particular items inside it is a later step.", "open_folder_only")
    if re.search(r'"\+"|\bFAB\b|"Text" option|"Event" option', first, re.I) and re.search(r"new|add|creat|entry form", lower):
        kind_match = re.search(r"(?:new|add(?:ing)?|creat(?:e|ing)) (?:a |an |new |text-based |text )*(expense|recipe|note|event|folder)", lower)
        if not kind_match:
            kind_match = re.search(r"(expense|recipe|note|event|folder) (?:entry )?(?:form|editor|dialog)", lower)
        if kind_match:
            entity = kind_match.group(1)
            return (f"The add/create control for a new {entity} is visible and enabled.",
                    f"A form, editor, or creation dialog for a new {entity} opens. Filling its fields and completing the save can require later actions.", "open_creation_form")
    tab = re.search(r'["`]([^"`]{1,70})["`] (?:tab|section|option)', description)
    if tab and re.search(r"switching the view|bottom navigation|specifications|overview|wallpaper|my photos|rewards & offers", first):
        name = json.dumps(tab.group(1), ensure_ascii=False)
        return (f"The {name} navigation entry is visible and can be selected.",
                f"The {name} section becomes visible for inspection. Reading its details or carrying out further operations within it is a later step.", "open_named_section")
    post_title = re.search(r'post titled ["`]([^"`]+)["`]', description, re.I)
    if post_title:
        title = json.dumps(post_title.group(1), ensure_ascii=False)
        return (f"The post {title} is visible and selectable in the current feed or result grid.",
                f"The selected post {title} opens so its content can be inspected. Saving, sharing, or summarizing it can require later actions.", "open_named_post")
    direction = re.search(r'["`](Right|Left|Up|Down)["`] button', description, re.I)
    if direction and '"x"' in lower:
        name = direction.group(1).lower()
        return (f"The grid's {name} control is available and the X marker has an allowed neighboring cell in that direction.",
                f"The X marker moves one cell {name}. Reaching the final target cell can require additional button presses.", "single_grid_move")
    if re.search(r'"Install" button', description, re.I):
        return ("The requested app's store listing is open, installation is allowed, and its Install control is enabled.",
                "The requested app begins downloading or installing, or an installation confirmation becomes available. Finishing installation and launching the app are later steps.", "start_installation")
    if re.search(r'"Add to Cart" button', description, re.I):
        return ("The intended product and variant are selected and their Add to Cart control is available.",
                "The intended product is added to the cart, with a cart update or confirmation reflecting that item. Checkout and payment are not part of this tap.", "add_requested_item_to_cart")
    if re.search(r'"Next Frame" button', description, re.I):
        return ("The requested video is loaded in VLC and the Next Frame control is available.",
                "The video advances by one frame for inspection. Transcribing all remaining frames is a later task.", "single_video_frame")
    if "date picker" in lower and ('">"' in lower or "next month" in lower):
        return ("The date picker is open and its next-month control is enabled.",
                "The date picker advances by one month. Selecting the final event date is a separate action.", "single_calendar_month")
    if re.search(r'"Add new folder" option', description, re.I):
        return ("The Add new folder option is available in the current menu.",
                "A new-folder creation dialog opens. Naming the folder, confirming it, or creating a track are separate operations.", "open_folder_creation_dialog")
    effect, target = explicit_tap_effect(description), tap_target(description)
    if effect and target:
        if re.match(r"delete|remove", effect, re.I) and "confirmation dialog" in lower:
            effect += ", or present the confirmation dialog for that deletion"
        return (f"The intended tap target, {target}, must be visible and operable.",
                f"The tap should {effect.rstrip('.')} for the intended target. An equivalent effect on a different item does not satisfy this step.", "explicit_action_effect")
    return None


def validate_output(source, output, protected):
    """Re-read artifacts and check invariants independently of rewrite loops."""
    changes = [json.loads(x) for x in (output / "semantic_changes.jsonl").read_text().splitlines()]
    by_row = {}
    for change in changes:
        by_row.setdefault((change["split"], change["line"]), []).append(change)
    report, image_count, reason_changes = {}, 0, []
    for split, expected_count in (("train", 882), ("val", 71), ("test", 1082)):
        before = [json.loads(x) for x in (source / f"{split}.jsonl").read_text().splitlines()]
        after = [json.loads(x) for x in (output / f"{split}.jsonl").read_text().splitlines()]
        refs = [json.loads(x) for x in (output / f"{split}.references.jsonl").read_text().splitlines()]
        old_refs = [json.loads(x) for x in (source / f"{split}.references.jsonl").read_text().splitlines()]
        assert len(before) == len(after) == len(refs) == len(old_refs) == expected_count
        assert (source / f"{split}.inputs.jsonl").read_bytes() == (output / f"{split}.inputs.jsonl").read_bytes()
        labels, statuses, changed_rows = Counter(), Counter(), 0
        for line, (old, new, ref, old_ref) in enumerate(zip(before, after, refs, old_refs), 1):
            old_target = json.loads(old["messages"][1]["content"][0]["text"])
            target = json.loads(new["messages"][1]["content"][0]["text"])
            assert list(target) == ["precondition", "postcondition", "reason", "answer"]
            assert all(isinstance(v, str) and v.strip() for v in target.values())
            assert target["answer"] == old_target["answer"]
            assert target["answer"] in ("Yes", "No")
            labels[target["answer"]] += 1
            statuses[ref["condition_review_status"]] += 1
            if target["reason"] != old_target["reason"]:
                reason_changes.append([split, line])
            restored = copy.deepcopy(new)
            restored["messages"][1]["content"][0]["text"] = old["messages"][1]["content"][0]["text"]
            assert restored == old, (split, line, "non-target field changed")
            recovered_target = copy.deepcopy(target)
            for change in by_row.get((split, line), []):
                assert recovered_target[change["field"]] == change["new"]
                recovered_target[change["field"]] = change["old"]
            assert recovered_target == old_target, (split, line, "incomplete change log")
            changed_rows += target != old_target
            assert ref["original_sample"] == old_ref["original_sample"]
            if old_ref["condition_review_status"] == "previously_reviewed_initial_transition":
                assert target == old_target, (split, line, "previous visual review changed")
            for content in new["messages"][0]["content"]:
                if content["type"] == "image":
                    relative = content["image"]
                    assert (output / relative).is_file()
                    assert (output / relative).samefile(source / relative)
                    image_count += 1
        report[split] = {"rows": len(after), "labels": dict(labels), "changed_rows": changed_rows,
                         "condition_review_status": dict(statuses)}
    assert reason_changes == [["train", 1]]
    assert image_count == 4070
    assert all(digest(Path(p)) == expected for p, expected in protected.items())
    assert (source / "prompt_template.txt").read_bytes() == (output / "prompt_template.txt").read_bytes()
    return {"passed": True, "splits": report, "all_source_hashes_unchanged": True,
            "all_input_messages_unchanged": True, "labels_unchanged": True,
            "reason_changes": reason_changes, "change_log_reversible": True,
            "original_samples_preserved": True, "previously_reviewed_pairs_preserved": True,
            "image_references": image_count, "all_images_same_original_files": True,
            "copied_image_files": 0, "scope": "Structural and provenance validation; not exhaustive semantic/visual validation."}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    metadata = json.loads((source / "metadata.json").read_text())
    assert metadata["schema_version"] == "qreflector_condition_generation_v1"
    source_split_statistics = copy.deepcopy(metadata["splits"])
    protected = {str(p): digest(p) for p in source.iterdir() if p.is_file()}
    protected.update(metadata["protected_source_sha256"])
    original_training = Path(__file__).resolve().parents[1] / '../training/SFT/training_data_64'
    for filename in ("train.jsonl", "val.jsonl"):
        p = original_training / filename
        protected[str(p)] = digest(p)
    products, changes, pending, decisions, stats = {}, [], [], [], {}
    for split in ("train", "val", "test"):
        lines = (source / f"{split}.jsonl").read_text().splitlines(keepends=True)
        refs = [json.loads(line) for line in (source / f"{split}.references.jsonl").read_text().splitlines()]
        assert len(lines) == len(refs) == {"train": 882, "val": 71, "test": 1082}[split]
        result, new_refs, categories, counts = [], [], Counter(), Counter()
        for line, (raw, ref) in enumerate(zip(lines, refs), 1):
            row = json.loads(raw)
            context = context_of(row)
            before = json.loads(row["messages"][1]["content"][0]["text"])
            target = copy.deepcopy(before)
            identity = (split, line)
            is_corrupt = any(Path(p).name.startswith("corrupt_") for p in ref["source_images"])
            proposal = MANUAL.get(identity)
            if proposal is None and ref["condition_review_status"] != "previously_reviewed_initial_transition":
                proposal = propose(context, is_corrupt)
            if proposal:
                target["precondition"], target["postcondition"], rule = proposal
                status = "visually_reviewed" if rule.startswith("visually_reviewed") else "rule_or_text_revised"
                categories[rule] += 1
            elif ref["condition_review_status"] == "previously_reviewed_initial_transition":
                rule, status = "retain_previously_reviewed_pair", "previously_reviewed"
            else:
                rule, status = "no_supported_automatic_repair", "pending_review"
                pending.append({"split": split, "line": line, "issue": "Condition semantics not established by a supported repair rule.",
                                "context": context, "current_target": before})
            if identity == ("train", 1):
                target["reason"] = MAIL_REASON
            if identity in KNOWN_ISSUES:
                pending.append({"split": split, "line": line, "issue": KNOWN_ISSUES[identity],
                                "context": context, "current_target": target})
            row_changes = []
            for field in ("precondition", "postcondition", "reason"):
                if target[field] != before[field]:
                    change = {"split": split, "line": line, "field": field, "old": before[field], "new": target[field], "rule": rule}
                    changes.append(change); row_changes.append(field)
            assert target["answer"] == before["answer"]
            if row_changes:
                row["messages"][1]["content"][0]["text"] = json.dumps(target, ensure_ascii=False)
                result.append(json.dumps(row, ensure_ascii=False) + "\n")
                counts["changed_rows"] += 1
            else:
                result.append(raw)
            counts[status] += 1
            counts["rows"] += 1
            decision = {"split": split, "line": line, "status": status, "rule": rule,
                        "changed_fields": row_changes, "additional_review_issue": KNOWN_ISSUES.get(identity)}
            decisions.append(decision)
            ref["previous_condition_review_status"] = ref["condition_review_status"]
            ref["condition_review_status"] = status
            ref["semantic_repair"] = decision
            new_refs.append(ref)
        assert len(lines) == len(refs)
        products[f"{split}.jsonl"] = "".join(result)
        products[f"{split}.references.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_refs)
        products[f"{split}.inputs.jsonl"] = (source / f"{split}.inputs.jsonl").read_text()
        stats[split] = {**dict(counts), "rules": dict(categories)}
    assert all(digest(Path(p)) == expected for p, expected in protected.items())
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for p in (source / "images").iterdir():
        (output / "images" / p.name).symlink_to(os.path.relpath(p.resolve(), output / "images"), target_is_directory=True)
    for name, contents in products.items():
        (output / name).write_text(contents)
    for name, records in [("semantic_changes.jsonl", changes), ("semantic_decisions.jsonl", decisions), ("review_required.jsonl", pending)]:
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "prompt_template.txt").write_bytes((source / "prompt_template.txt").read_bytes())
    validation = validate_output(source, output, protected)
    (output / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    for split in ("train", "val", "test"):
        metadata["splits"][split]["condition_review_status"] = validation["splits"][split]["condition_review_status"]
    metadata["previous_version_split_statistics"] = source_split_statistics
    metadata["previous_version_review_examples"] = metadata.pop("review_examples", [])
    metadata.update({
        "dataset_version": "condition_generation_v2_semantic_pass1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_version_directory": str(source),
        "semantic_repair_script": str(Path(__file__).resolve()),
        "semantic_repair_script_sha256": digest(Path(__file__)),
        "semantic_reannotation_performed": True,
        "semantic_repair_statistics": stats,
        "semantic_repair_scope": "Conditions with an explicit supported action intent; one visually incorrect training reason corrected; no label, context, action, image, split or order changes.",
        "condition_mapping": "Latest conditions follow semantic_decisions/semantic_changes. Per-split references preserve the legacy source mappings and original samples for provenance; do not treat those legacy mappings as the current values.",
        "protected_source_sha256": protected,
        "label_policy": "All Yes/No labels preserved from v1. The train:1 reason was corrected from direct image inspection; all other reasons preserved.",
        "condition_targets_ready_for_unreviewed_training": False,
        "pending_review_records": len(pending),
        "pending_review_unique_rows": len({(r["split"], r["line"]) for r in pending}),
        "pending_condition_rows": sum(s.get("pending_review", 0) for s in stats.values()),
        "limitations": [
            "Rule-based revisions are not equivalent to exhaustive human/visual annotation; remaining unhandled or ambiguous cases are listed explicitly.",
            "Labels are frozen, including test labels. Condition repair alone does not resolve all legacy verdict/reason/subgoal inconsistencies.",
            "Before/after images were visually inspected in this pass only for train:1, train:25, test:40; earlier reviewed pairs retain their prior provenance.",
        ],
    })
    metadata.pop("output_sha256", None)
    metadata["output_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    assert all(digest(Path(p)) == expected for p, expected in protected.items())
    print(json.dumps({"output": str(output), "splits": stats, "changed_fields": len(changes), "pending_records": len(pending)}, indent=2))


if __name__ == "__main__":
    main()
