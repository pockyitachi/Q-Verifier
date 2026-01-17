import time
import os
from Agents.base_agent import BaseAgent
from Agents.info_pool import InfoPool
from constant import ATOMIC_ACTION_SIGNATURES
from AgentsUtils.ocr_utils import run_ocr
from AgentsUtils.tool_utils import tap, type, swipe, enter, switch_app, back, home, save_screenshot_to_file
from AgentsUtils.common_utils import extract_json_object


class Actor(BaseAgent):
    def __init__(self, adb_path):
        self.adb = adb_path

    def init_chat(self):
        operation_history = []
        system_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the user's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def init_chat_recovery(self):
        operation_history = []
        system_prompt = "You are a helpful AI assistant for operating mobile phones. Your goal is to choose the correct actions to complete the recovery plan's instruction. Think as if you are a human user operating the phone."
        operation_history.append(["system", [{"type": "text", "text": system_prompt}]])
        return operation_history

    def get_prompt(self, info_pool: InfoPool) -> str:
        prompt = "### User Instruction ###\n"
        prompt += f"{info_pool.instruction}\n\n"

        prompt += "### Overall Plan ###\n"
        prompt += f"{info_pool.plan}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subtask ###\n"
        prompt += f"{info_pool.current_subgoal}\n\n"

        prompt += "### Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better understand the content in this screenshot, we have extracted positional information for the text elements and icons, including interactive elements such as search bars. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom)."
        )
        prompt += "The extracted information is as follows:\n"

        for clickable_info in info_pool.perception_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info[
                'coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += (
            "Note that a search bar is often a long, rounded rectangle. If no search bar is presented and you want to perform a search, you may need to tap a search button, which is commonly represented by a magnifying glass.\n"
            "IMPORTANT: Elements with y-coordinate less than ~150 pixels are typically in the STATUS BAR area (battery, signal, time, etc.) which is NOT interactive. "
            "When tapping UI elements like buttons or icons, make sure to use coordinates from the ACTUAL app content area (usually y > 150).\n"
            "Also, the information above might not be entirely accurate. "
            "You should combine it with the screenshot to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Keyboard status ###\n"
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        if info_pool.tips != "":
            prompt += "### Tips ###\n"
            prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
            prompt += f"{info_pool.tips}\n\n"

        prompt += "### Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += "Here are some potentially important content relevant to the user's request you already recorded:\n"
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "---\n"
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions or the shortcuts. The shortcuts are predefined sequences of actions that can be used to speed up the process. Each shortcut has a precondition specifying when it is suitable to use. If you plan to use a shortcut, ensure the current phone state satisfies its precondition first.\n\n"

        prompt += "#### Atomic Actions ####\n"
        prompt += "The atomic action functions are listed in the format of `name(arguments): description` as follows:\n"

        if info_pool.keyboard_pre:
            for action, value in ATOMIC_ACTION_SIGNATURES.items():
                prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        else:
            for action, value in ATOMIC_ACTION_SIGNATURES.items():
                if "Type" not in action:
                    prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
            prompt += "NOTE: Unable to type. The keyboard has not been activated. To type, please activate the keyboard by tapping on an input box or using a shortcut, which includes tapping on an input box first.”\n"

        prompt += "\n"
        prompt += "#### Shortcuts ####\n"
        if info_pool.shortcuts != {}:
            prompt += "The shortcut functions are listed in the format of `name(arguments): description | Precondition: precondition` as follows:\n"
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"- {shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are available.\n"
        prompt += "\n"

        prompt += "### Latest Action History ###\n"
        if info_pool.action_history != []:
            prompt += "Recent actions you took previously and whether they were successful:\n"
            num_actions = min(5, len(info_pool.action_history))
            latest_actions = info_pool.action_history[-num_actions:]
            latest_summary = info_pool.summary_history[-num_actions:]
            latest_outcomes = info_pool.action_outcomes[-num_actions:]
            error_descriptions = info_pool.error_descriptions[-num_actions:]
            action_log_strs = []
            for act, summ, outcome, err_des in zip(latest_actions, latest_summary, latest_outcomes, error_descriptions):
                if outcome == "A":
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Successful\n"
                else:
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"
                prompt += action_log_str
                action_log_strs.append(action_log_str)
            if latest_outcomes[-1] == "C" and "Tap" in action_log_strs[-1] and "Tap" in action_log_strs[-2]:
                prompt += "\nHINT: If multiple Tap actions failed to make changes to the screen, consider using a \"Swipe\" action to view more content or use another way to achieve the current subgoal."

            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "---\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        prompt += "Provide a detailed explanation of your rationale for the chosen action. IMPORTANT: If you decide to use a shortcut, first verify that its precondition is met in the current phone state. For example, if the shortcut requires the phone to be at the Home screen, check whether the current screenshot shows the Home screen. If not, perform the appropriate atomic actions instead.\n\n"

        prompt += "### Action ###\n"
        prompt += "Choose only one action or shortcut from the options provided. IMPORTANT: Do NOT return invalid actions like null or stop. Do NOT repeat previously failed actions.\n"
        prompt += "Use shortcuts whenever possible to expedite the process, but make sure that the precondition is met.\n"
        prompt += "You must provide your decision using a valid JSON format specifying the name and arguments of the action. For example, if you choose to tap at position (100, 200), you should write {\"name\":\"Tap\", \"arguments\":{\"x\":100, \"y\":100}}. If an action does not require arguments, such as Home, fill in null to the \"arguments\" field. Ensure that the argument keys match the action function's signature exactly.\n\n"

        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action and the expected outcome."
        return prompt

    def get_prompt_recovery(self, info_pool: InfoPool) -> str:
        recovery_plan_dict = info_pool.recovery_plan
        recovery_plan = recovery_plan_dict.get(info_pool.recovery_target_state_id, {})
        current_sub_task = recovery_plan["current_subgoal"]
        if not recovery_plan:
            print("Recovery plan is empty. Skipping execution.")
            info_pool.state_agent_recovery_mode = False
            return ""
        recovery_goal = recovery_plan["recovery_goal"]
        prompt = "### Recovery Goal ###\n"
        prompt += f"{recovery_goal}\n\n"

        prompt += "### Overall Recovery Plan ###\n"
        prompt += f"{recovery_plan}\n\n"

        prompt += "### Progress Status ###\n"
        if info_pool.progress_status != "":
            prompt += f"{info_pool.progress_status}\n\n"
        else:
            prompt += "No progress yet.\n\n"

        prompt += "### Current Subtask ###\n"
        prompt += f"{current_sub_task}\n\n"

        prompt += "### Screen Information ###\n"
        prompt += (
            f"The attached image is a screenshot showing the current state of the phone. "
            f"Its width and height are {info_pool.width} and {info_pool.height} pixels, respectively.\n"
        )
        prompt += (
            "To help you better understand the content in this screenshot, we have extracted positional information for the text elements and icons, including interactive elements such as search bars. "
            "The format is: (coordinates; content). The coordinates are [x, y], where x represents the horizontal pixel position (from left to right) "
            "and y represents the vertical pixel position (from top to bottom)."
        )
        prompt += "The extracted information is as follows:\n"

        for clickable_info in info_pool.perception_infos_pre:
            if clickable_info['text'] != "" and clickable_info['text'] != "icon: None" and clickable_info[
                'coordinates'] != (0, 0):
                prompt += f"{clickable_info['coordinates']}; {clickable_info['text']}\n"
        prompt += "\n"
        prompt += (
            "Note that a search bar is often a long, rounded rectangle. If no search bar is presented and you want to perform a search, you may need to tap a search button, which is commonly represented by a magnifying glass.\n"
            "IMPORTANT: Elements with y-coordinate less than ~150 pixels are typically in the STATUS BAR area (battery, signal, time, etc.) which is NOT interactive. "
            "When tapping UI elements like buttons or icons, make sure to use coordinates from the ACTUAL app content area (usually y > 150).\n"
            "Also, the information above might not be entirely accurate. "
            "You should combine it with the screenshot to gain a better understanding."
        )
        prompt += "\n\n"

        prompt += "### Keyboard status ###\n"
        if info_pool.keyboard_pre:
            prompt += "The keyboard has been activated and you can type."
        else:
            prompt += "The keyboard has not been activated and you can\'t type."
        prompt += "\n\n"

        if info_pool.tips != "":
            prompt += "### Tips ###\n"
            prompt += "From previous experience interacting with the device, you have collected the following tips that might be useful for deciding what to do next:\n"
            prompt += f"{info_pool.tips}\n\n"

        prompt += "### Important Notes ###\n"
        if info_pool.important_notes != "":
            prompt += "Here are some potentially important content relevant to the user's request you already recorded:\n"
            prompt += f"{info_pool.important_notes}\n\n"
        else:
            prompt += "No important notes recorded.\n\n"

        prompt += "---\n"
        prompt += "Carefully examine all the information provided above and decide on the next action to perform. If you notice an unsolved error in the previous action, think as a human user and attempt to rectify them. You must choose your action from one of the atomic actions or the shortcuts. The shortcuts are predefined sequences of actions that can be used to speed up the process. Each shortcut has a precondition specifying when it is suitable to use. If you plan to use a shortcut, ensure the current phone state satisfies its precondition first.\n\n"

        prompt += "#### Atomic Actions ####\n"
        prompt += "The atomic action functions are listed in the format of `name(arguments): description` as follows:\n"

        if info_pool.keyboard_pre:
            for action, value in ATOMIC_ACTION_SIGNATURES.items():
                prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
        else:
            for action, value in ATOMIC_ACTION_SIGNATURES.items():
                if "Type" not in action:
                    prompt += f"- {action}({', '.join(value['arguments'])}): {value['description'](info_pool)}\n"
            prompt += "NOTE: Unable to type. The keyboard has not been activated. To type, please activate the keyboard by tapping on an input box or using a shortcut, which includes tapping on an input box first.”\n"

        prompt += "\n"
        prompt += "#### Shortcuts ####\n"
        if info_pool.shortcuts != {}:
            prompt += "The shortcut functions are listed in the format of `name(arguments): description | Precondition: precondition` as follows:\n"
            for shortcut, value in info_pool.shortcuts.items():
                prompt += f"- {shortcut}({', '.join(value['arguments'])}): {value['description']} | Precondition: {value['precondition']}\n"
        else:
            prompt += "No shortcuts are available.\n"
        prompt += "\n"

        prompt += "### Latest Action History ###\n"
        if info_pool.action_history != []:
            prompt += "Recent actions you took previously and whether they were successful:\n"
            num_actions = min(5, len(info_pool.action_history))
            latest_actions = info_pool.action_history[-num_actions:]
            latest_summary = info_pool.summary_history[-num_actions:]
            latest_outcomes = info_pool.action_outcomes[-num_actions:]
            error_descriptions = info_pool.error_descriptions[-num_actions:]
            action_log_strs = []
            for act, summ, outcome, err_des in zip(latest_actions, latest_summary, latest_outcomes, error_descriptions):
                if outcome == "A":
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Successful\n"
                else:
                    action_log_str = f"Action: {act} | Description: {summ} | Outcome: Failed | Feedback: {err_des}\n"
                prompt += action_log_str
                action_log_strs.append(action_log_str)
            if latest_outcomes[-1] == "C" and "Tap" in action_log_strs[-1] and "Tap" in action_log_strs[-2]:
                prompt += "\nHINT: If multiple Tap actions failed to make changes to the screen, consider using a \"Swipe\" action to view more content or use another way to achieve the current subgoal."

            prompt += "\n"
        else:
            prompt += "No actions have been taken yet.\n\n"

        prompt += "---\n"
        prompt += "Provide your output in the following format, which contains three parts:\n"
        prompt += "### Thought ###\n"
        prompt += "Provide a detailed explanation of your rationale for the chosen action. IMPORTANT: If you decide to use a shortcut, first verify that its precondition is met in the current phone state. For example, if the shortcut requires the phone to be at the Home screen, check whether the current screenshot shows the Home screen. If not, perform the appropriate atomic actions instead.\n\n"

        prompt += "### Action ###\n"
        prompt += "Choose only one action or shortcut from the options provided. IMPORTANT: Do NOT return invalid actions like null or stop. Do NOT repeat previously failed actions.\n"
        prompt += "Use shortcuts whenever possible to expedite the process, but make sure that the precondition is met.\n"
        prompt += "You must provide your decision using a valid JSON format specifying the name and arguments of the action. For example, if you choose to tap at position (100, 200), you should write {\"name\":\"Tap\", \"arguments\":{\"x\":100, \"y\":100}}. If an action does not require arguments, such as Home, fill in null to the \"arguments\" field. Ensure that the argument keys match the action function's signature exactly.\n\n"

        prompt += "### Description ###\n"
        prompt += "A brief description of the chosen action and the expected outcome."
        return prompt

    def _validate_shortcut_arguments(self, shortcut: dict, provided_arguments: dict) -> str:
        """
        Validate that all placeholders in a shortcut's arguments_map can be resolved.
        
        Args:
            shortcut: The shortcut definition dict
            provided_arguments: The arguments provided when calling the shortcut
            
        Returns:
            str: Error message if validation fails, None if valid
        """
        shortcut_name = shortcut.get("name", "Unknown")
        
        for i, atomic_action in enumerate(shortcut.get("atomic_action_sequence", [])):
            arguments_map = atomic_action.get("arguments_map", {})
            if not arguments_map:
                continue
                
            for arg_key, mapped_value in arguments_map.items():
                if isinstance(mapped_value, str):
                    # Check if it's a numeric string (hardcoded coordinate)
                    try:
                        float(mapped_value)
                        # It's a valid number string, OK
                        continue
                    except ValueError:
                        pass
                    
                    # It's a placeholder string - check if it can be resolved
                    if provided_arguments and mapped_value in provided_arguments:
                        # Placeholder is in provided arguments, OK
                        continue
                    
                    # Unresolvable placeholder!
                    return (f"Step {i} ({atomic_action.get('name')}): placeholder '{mapped_value}' "
                            f"for argument '{arg_key}' is not in provided arguments {list(provided_arguments.keys()) if provided_arguments else []}")
        
        return None  # Validation passed

    def execute_atomic_action(self, action: str, arguments: dict, **kwargs) -> None:
        adb_path = self.adb

        if "Open_App".lower() == action.lower():
            screenshot_file = kwargs["screenshot_file"]
            ocr_detection = kwargs["ocr_detection"]
            ocr_recognition = kwargs["ocr_recognition"]
            app_name = arguments["app_name"].strip()
            app_name_lower = app_name.lower()
            
            # App name to package mapping for ADB fallback
            APP_PACKAGES = {
                "simple sms messenger": "com.simplemobiletools.smsmessenger/.activities.MainActivity",
                "sms messenger": "com.simplemobiletools.smsmessenger/.activities.MainActivity",
                "simple calendar": "com.simplemobiletools.calendar.pro/.activities.MainActivity",
                "calendar": "com.simplemobiletools.calendar.pro/.activities.MainActivity",
                "simple gallery": "com.simplemobiletools.gallery.pro/.activities.MainActivity",
                "gallery": "com.simplemobiletools.gallery.pro/.activities.MainActivity",
                "simple draw": "com.simplemobiletools.draw.pro/.activities.MainActivity",
                "markor": "net.gsantner.markor/.activity.MainActivity",
                "vlc": "org.videolan.vlc/.StartActivity",
                "camera": "com.android.camera2/com.android.camera.CameraLauncher",
                "clock": "com.google.android.deskclock/com.android.deskclock.DeskClock",
                "contacts": "com.simplemobiletools.contacts.pro/.activities.MainActivity",
                "simple contacts": "com.simplemobiletools.contacts.pro/.activities.MainActivity",
                "broccoli": "com.flauschcode.broccoli/.activities.MainActivity",
                "recipe": "com.flauschcode.broccoli/.activities.MainActivity",
                "tasks": "org.tasks/.activities.MainActivity",
                "expense": "com.arduia.expense/.MainActivity",
                "retro music": "code.name.monkey.retromusic/.activities.MainActivity",
                "osmand": "net.osmand/.plus.activities.MapActivity",
            }
            
            # Try OCR first - but require meaningful match (min 3 chars)
            found_via_ocr = False
            text, coordinate = run_ocr(screenshot_file, ocr_detection, ocr_recognition)
            for ti in range(len(text)):
                ocr_text = text[ti].lower().strip()
                # Skip very short OCR texts (likely noise)
                if len(ocr_text) < 3:
                    continue
                # Check for meaningful match - OCR text should be substantial part of app name
                # or app name should contain the full OCR text (for truncated labels like "SMS Messe...")
                if (app_name_lower in ocr_text) or (ocr_text in app_name_lower and len(ocr_text) >= 4):
                    # Calculate center of bounding box
                    center_x = int((coordinate[ti][0] + coordinate[ti][2]) / 2)
                    center_y = int((coordinate[ti][1] + coordinate[ti][3]) / 2)
                    # Tap at center of the detected text
                    tap(adb_path, center_x, center_y)
                    found_via_ocr = True
                    print(f"[Open_App] Found '{text[ti]}' via OCR, tapping at ({center_x}, {center_y})")
                    break
            
            # ADB fallback if OCR didn't find the app
            if not found_via_ocr:
                package = None
                for key, pkg in APP_PACKAGES.items():
                    if key in app_name_lower or app_name_lower in key:
                        package = pkg
                        break
                
                if package:
                    print(f"[Open_App] OCR failed, using ADB fallback: {package}")
                    import subprocess
                    subprocess.run([adb_path, "shell", "am", "start", "-n", package], 
                                   capture_output=True, timeout=10)
                else:
                    print(f"[Open_App] Warning: Could not find app '{app_name}' via OCR or package mapping")
            
            if app_name in ['Fandango', 'Walmart', 'Best Buy']:
                # additional wait time for app loading
                time.sleep(10)
            time.sleep(10)

        elif "Tap".lower() == action.lower():
            tap(adb_path, int(arguments["x"]), int(arguments["y"]))
            time.sleep(5)

        elif "Swipe".lower() == action.lower():
            swipe(adb_path, int(arguments["x1"]), int(arguments["y1"]), int(arguments["x2"]), int(arguments["y2"]))
            time.sleep(5)

        elif "Type".lower() == action.lower():
            type(adb_path, arguments["text"])
            time.sleep(3)

        elif "Enter".lower() == action.lower():
            enter(adb_path)
            time.sleep(10)

        elif "Back".lower() == action.lower():
            back(adb_path)
            time.sleep(3)

        elif "Home".lower() == action.lower():
            home(adb_path)
            time.sleep(3)

        elif "Switch_App".lower() == action.lower():
            switch_app(adb_path)
            time.sleep(3)

        elif "Wait".lower() == action.lower():
            time.sleep(10)

    def execute(self, action_str: str, info_pool: InfoPool, screenshot_log_dir=None, iter="", **kwargs) -> None:
        action_object = extract_json_object(action_str)
        if action_object is None:
            print("Error! Invalid JSON for executing action: ", action_str)
            return None, 0, None
        action, arguments = action_object["name"].strip(), action_object["arguments"]

        if action in ATOMIC_ACTION_SIGNATURES:
            print("Executing atomic action: ", action, arguments)
            self.execute_atomic_action(action, arguments, info_pool=info_pool, **kwargs)
            if screenshot_log_dir:
                time.sleep(1)
                screenshot_file = os.path.join(screenshot_log_dir, f"{iter}__{action.replace(' ', '')}.png")
                save_screenshot_to_file(self.adb, screenshot_file)
            return action_object, 1, None
        elif action in info_pool.shortcuts:
            print("Executing shortcut: ", action)
            shortcut = info_pool.shortcuts[action]
            
            # Validate shortcut before execution - check all placeholders can be resolved
            validation_error = self._validate_shortcut_arguments(shortcut, arguments)
            if validation_error:
                print(f"[Actor] WARNING: Shortcut '{action}' is invalid: {validation_error}")
                print(f"[Actor] Skipping shortcut. This shortcut has unresolvable placeholders.")
                # Return error but don't crash - allow the system to continue
                error_msg = f"InvalidShortcut: {validation_error}"
                return action_object, 0, error_msg
            
            for i, atomic_action in enumerate(shortcut["atomic_action_sequence"]):
                try:
                    atomic_action_name = atomic_action["name"]
                    if atomic_action["arguments_map"] is None or len(atomic_action["arguments_map"]) == 0:
                        atomic_action_args = None
                    else:
                        atomic_action_args = {}
                        for atomic_arg_key, value in atomic_action["arguments_map"].items():
                            if value in arguments:  # if the mapped key is in the shortcut arguments
                                atomic_action_args[atomic_arg_key] = arguments[value]
                            else:  # if not: the values are directly passed
                                atomic_action_args[atomic_arg_key] = value
                    print(f"\t Executing sub-step {i}:", atomic_action_name, atomic_action_args, "...")
                    self.execute_atomic_action(atomic_action_name, atomic_action_args, info_pool=info_pool, **kwargs)
                    if screenshot_log_dir is not None:
                        time.sleep(1)
                        screenshot_file = os.path.join(screenshot_log_dir,
                                                       f"{iter}__{action.replace(' ', '')}__{i}-{atomic_action_name.replace(' ', '')}.png")
                        save_screenshot_to_file(self.adb, screenshot_file)
                except Exception as e:
                    error_msg = f"Error in executing step {i}: {atomic_action_name} {atomic_action_args} - {str(e)}"
                    print("Error in executing shortcut: ", action, error_msg)
                    return action_object, i, error_msg
            return action_object, len(shortcut["atomic_action_sequence"]), None

        else:
            if action.lower() in ["null", "none", "finish", "exit", "stop"]:
                print("Agent choose to finish the task. Action: ", action)
            else:
                print("Error! Invalid action name: ", action)
            info_pool.finish_thought = info_pool.last_action_thought
            return None, 0, None

    def parse_response(self, response: str) -> dict:
        thought = response.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace("  ",
                                                                                                              " ").strip()
        action = response.split("### Action ###")[-1].split("### Description ###")[0].replace("\n", " ").replace("  ",
                                                                                                                 " ").strip()
        description = response.split("### Description ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        return {"thought": thought, "action": action, "description": description}
