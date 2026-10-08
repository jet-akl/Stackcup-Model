import sys
from typing import Dict, List, Tuple
import pygame

class BaseXbox():
    def __init__(
            self,
            joy_id=0,
            axis_deadzone=0.15,
            fps=15,
            ):
        self.BUTTON_NAME_MAP = {
            0: "A",
            1: "B",
            2: "X",
            3: "Y",
            4: "LB",
            5: "RB",
            6: "BACK",
            7: "START",
            8: "L_STICK",
            9: "R_STICK",
            10: "GUIDE",
        }

        self.AXIS_NAME_MAP = {
            0: "LX",
            1: "LY",
            2: "LT",
            3: "RX",
            4: "RY",
            5: "RT",
        }

        self.joy_id = joy_id
        self.axis_deadzone = axis_deadzone
        self.fps = fps

        self.joy = None
        self.connect() 

    def connect(self,joy_id=None):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("Error : No controller detected. Please connect an Xbox controller and retry.")
            self.joy = None
            return 0
        else:
            try:
                self.joy_id = joy_id if joy_id is not None else self.joy_id
                self.joy = pygame.joystick.Joystick(self.joy_id)
                self.joy.init()
                print(f'{"=" * 15} Successful Connection {"=" * 15}')
                print(f"Controller index: {self.joy.get_instance_id()}")
                print(f"Name: {self.joy.get_name()}")
                print(f"GUID: {self.joy.get_guid()}")
                print(f"Buttons: {self.joy.get_numbuttons()}")
                print(f"Axes: {self.joy.get_numaxes()}")
                print(f"Hats: {self.joy.get_numhats()}")
                print("=" * 53)
            except:
                print(f"Error : Could not connect device:{self.joy_id}.Please check again.")

    def end(self):
        pygame.joystick.quit()
        pygame.quit()

    # 边缘触发
    def event_listening(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        
        _running = True
        try:
            while _running:
                for event in pygame.event.get():
                    if event.type == pygame.JOYDEVICEREMOVED and event.instance_id == self.joy_id:
                        print(f"\n[DISCONNECT] device:{self.joy_id} is out.")
                        _running = False
                    elif event.type == pygame.JOYBUTTONDOWN:
                        print(
                            f"[DOWN] button={self._button_name(event.button)}({event.button})"
                        )
                    elif event.type == pygame.JOYBUTTONUP:
                        print(
                            f"[UP] button={self._button_name(event.button)}({event.button})"
                        )
                    elif event.type == pygame.JOYAXISMOTION:
                        if abs(event.value) >= self.axis_deadzone:
                            print(
                                f"[AXIS] {self._axis_name(event.axis)}({event.axis})={event.value:+.3f}"
                            )
                    elif event.type == pygame.JOYHATMOTION:
                        print(
                            f"[HAT] hat={event.hat}, value={event.value}"
                        )
        except KeyboardInterrupt:
            self.end()
            print("\nGet Ctrl-c: Exit.")

    # 持续读取
    def keep_reading(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        self.connect()
        clock = pygame.time.Clock()
        try:
            while True:
                pygame.event.pump()
                
                pressed_buttons = self._get_pressed_buttons()
                active_axis = self._get_active_axis()
                pressed_hat = self._get_pressed_hat()
                if not pressed_buttons and not active_axis and not pressed_hat:
                    continue

                btn_txt = ", ".join(pressed_buttons) if pressed_buttons else "None"
                axis_txt = ", ".join(f"{name}={val:+.2f}" for name, val in active_axis) if active_axis else "None"
                hat_txt = ", ".join(f"{idx}={val}" for idx,val in pressed_hat) if pressed_hat else "None"

                print(f"[STATE] buttons: {btn_txt} | axes: {axis_txt} | hat: {hat_txt}")

                # update
                clock.tick(self.fps)

        except KeyboardInterrupt:
            self.end()
            print("\nGet Ctrl-c: Exit.")

    def _button_name(self,button_id):
        return self.BUTTON_NAME_MAP.get(button_id, f"No Button_Key")

    def _axis_name(self,axis_id):
        return self.AXIS_NAME_MAP.get(axis_id, f"No Axis_Key")
    
    def _get_pressed_buttons(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        
        pressed = []
        for i in range(self.joy.get_numbuttons()):
            if self.joy.get_button(i):
                pressed.append(self._button_name(i))
        return pressed
    
    def _get_active_axis(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        
        active = []
        for i in range(self.joy.get_numaxes()):
            value = self.joy.get_axis(i)
            if self._axis_name(i) in ('LT' , 'RT'): # LT/RT 从-1.0 -> 1.0
                if value >= -1.0 + self.axis_deadzone:
                    active.append((self._axis_name(i), value))
            elif abs(value) >= self.axis_deadzone:
                active.append((self._axis_name(i), value))
        return active

    def _get_pressed_hat(self):
        if self.joy is None:
            raise ValueError("joystick has not be inited.Please use 'connect' first.")
        
        pressed = []
        for i in range(self.joy.get_numhats()):
            x,y = self.joy.get_hat(i)
            if (x,y) != (0,0):
                pressed.append((i,(x,y)))
        return pressed

if __name__=='__main__':
    xbox = BaseXbox()
    # xbox.event_listening()
    xbox.keep_reading()