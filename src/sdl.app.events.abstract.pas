{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.events.abstract;

{$mode ObjFPC}{$H+}

{$optimization autoInline}

interface

uses sdl2, ctypes;

type
  { keyboard state }
  TCustomShiftStateEnum = (ssShift, ssAlt, ssCtrl,
    ssLeft, ssRight, ssUp, ssDown,
    ssSpace, ssEnter);

  TDragDropEvent = procedure(Sender, Source: TObject; X, Y: Integer) of object;

  TCustomShiftState = set of TCustomShiftStateEnum;
  TKeyEvent = procedure(Sender: TObject; var Key: Word; Shift: TCustomShiftStateEnum) of object;
  TKeyPressEvent = procedure(Sender: TObject; var Key: char) of object;

  { events }
  TOnQuitEvent = procedure(const event: TSDL_QuitEvent) of object;
  TOnWindowEventEvent = procedure(const event: TSDL_WindowEvent) of object;
  TOnWindowManagerEventEvent = procedure(const event: TSDL_SysWMEvent) of object;
  TOnKeyDownEvent = procedure(const event: TSDL_KeyboardEvent) of object;
  TOnKeyUpEvent = procedure(const event: TSDL_KeyboardEvent) of object;
  TOnTextEditingEvent = procedure(const event: TSDL_TextEditingEvent) of object;
  TOnTextInputEvent = procedure(const event: TSDL_TextInputEvent) of object;
  TOnMouseMotionEvent = procedure(const event: TSDL_MouseMotionEvent) of object;
  TOnMouseButtonDownEvent = procedure(const event: TSDL_MouseButtonEvent) of object;
  TOnMouseButtonUpEvent = procedure(const event: TSDL_MouseButtonEvent) of object;
  TOnMouseWheelEvent = procedure(const event: TSDL_MouseWheelEvent) of object;
  TOnJoyAxisMotionEvent = procedure(const event: TSDL_JoyAxisEvent) of object;
  TOnJoyBallMotionEvent = procedure(const event: TSDL_JoyBallEvent) of object;
  TOnJoyHatMotionEvent = procedure(const event: TSDL_JoyHatEvent) of object;
  TOnJoyButtonDownEvent = procedure(const event: TSDL_JoyButtonEvent) of object;
  TOnJoyButtonUpEvent = procedure(const event: TSDL_JoyButtonEvent) of object;
  TOnJoyDeviceAddedEvent = procedure(const event: TSDL_JoyDeviceEvent) of object;
  TOnJoyDeviceRemovedEvent = procedure(const event: TSDL_JoyDeviceEvent) of object;
  TOnControllerAxisMotionEvent = procedure(const event: TSDL_ControllerAxisEvent) of object;
  TOnControllerButtonDownEvent = procedure(const event: TSDL_ControllerButtonEvent) of object;
  TOnControllerButtonUpEvent = procedure(const event: TSDL_ControllerButtonEvent) of object;
  TOnControllerDeviceAddedEvent = procedure(const event: TSDL_ControllerDeviceEvent) of object;
  TOnControllerDeviceRemovedEvent = procedure(const event: TSDL_ControllerDeviceEvent) of object;
  TOnControllerDeviceRemappedEvent = procedure(const event: TSDL_ControllerDeviceEvent) of object;
  TOnControllerTouchPadMotionEvent = procedure(const event: TSDL_ControllerTouchpadEvent) of object;
  TOnControllerSensorUpdateEvent = procedure(const event: TSDL_ControllerSensorEvent) of object;
  TOnUserEvent = procedure(const event: TSDL_UserEvent) of object;

  { TEventHandler }

  TEventHandler = class
    private
      procedure HandleEvent(const event: TSDL_Event);
    private
      FOnControllerAxisMotion: TOnControllerAxisMotionEvent;
      FOnControllerButtonDown: TOnControllerButtonDownEvent;
      FOnControllerButtonUp: TOnControllerButtonUpEvent;
      FOnControllerDeviceAdded: TOnControllerDeviceAddedEvent;
      FOnControllerDeviceRemapped: TOnControllerDeviceRemappedEvent;
      FOnControllerDeviceRemoved: TOnControllerDeviceRemovedEvent;
      FOnControllerTouchPadMotion: TOnControllerTouchPadMotionEvent;
      FOnJoyAxisMotion: TOnJoyAxisMotionEvent;
      FOnJoyBallMotion: TOnJoyBallMotionEvent;
      FOnJoyButtonDown: TOnJoyButtonDownEvent;
      FOnJoyButtonUp: TOnJoyButtonUpEvent;
      FOnJoyDeviceAdded: TOnJoyDeviceAddedEvent;
      FOnJoyDeviceRemoved: TOnJoyDeviceRemovedEvent;
      FOnJoyHatMotion: TOnJoyHatMotionEvent;
      FOnKeyDown: TOnKeyDownEvent;
      FOnKeyUp: TOnKeyUpEvent;
      FOnMouseButtonDown: TOnMouseButtonDownEvent;
      FOnMouseButtonUp: TOnMouseButtonUpEvent;
      FOnMouseMotion: TOnMouseMotionEvent;
      FOnMouseWheel: TOnMouseWheelEvent;
      FOnQuit: TOnQuitEvent;
      FOnControllerSensorUpdateEvent: TOnControllerSensorUpdateEvent;
      FOnTextEditing: TOnTextEditingEvent;
      FOnTextInput: TOnTextInputEvent;
      FOnUserEvent: TOnUserEvent;
      FOnWindowEvent: TOnWindowEventEvent;
      FOnWindowManagerEvent: TOnWindowManagerEventEvent;
      procedure SetOnControllerAxisMotion(AValue: TOnControllerAxisMotionEvent);
      procedure SetOnControllerButtonDown(AValue: TOnControllerButtonDownEvent);
      procedure SetOnControllerButtonUp(AValue: TOnControllerButtonUpEvent);
      procedure SetOnControllerDeviceAdded(AValue: TOnControllerDeviceAddedEvent);
      procedure SetOnControllerDeviceRemapped(AValue: TOnControllerDeviceRemappedEvent);
      procedure SetOnControllerDeviceRemoved(AValue: TOnControllerDeviceRemovedEvent);
      procedure SetOnControllerTouchPadMotion(
        AValue: TOnControllerTouchPadMotionEvent);
      procedure SetOnJoyAxisMotion(AValue: TOnJoyAxisMotionEvent);
      procedure SetOnJoyBallMotion(AValue: TOnJoyBallMotionEvent);
      procedure SetOnJoyButtonDown(AValue: TOnJoyButtonDownEvent);
      procedure SetOnJoyButtonUp(AValue: TOnJoyButtonUpEvent);
      procedure SetOnJoyDeviceAdded(AValue: TOnJoyDeviceAddedEvent);
      procedure SetOnJoyDeviceRemoved(AValue: TOnJoyDeviceRemovedEvent);
      procedure SetOnJoyHatMotion(AValue: TOnJoyHatMotionEvent);
      procedure SetOnKeyDown(AValue: TOnKeyDownEvent);
      procedure SetOnKeyUp(AValue: TOnKeyUpEvent);
      procedure SetOnMouseButtonDown(AValue: TOnMouseButtonDownEvent);
      procedure SetOnMouseButtonUp(AValue: TOnMouseButtonUpEvent);
      procedure SetOnMouseMotion(AValue: TOnMouseMotionEvent);
      procedure SetOnMouseWheel(AValue: TOnMouseWheelEvent);
      procedure SetOnQuit(AValue: TOnQuitEvent);
      procedure SetOnControllerSensorUpdateEvent(AValue: TOnControllerSensorUpdateEvent);
      procedure SetOnTextEditing(AValue: TOnTextEditingEvent);
      procedure SetOnTextInput(AValue: TOnTextInputEvent);
      procedure SetOnUserEvent(AValue: TOnUserEvent);
      procedure SetOnWindowEvent(AValue: TOnWindowEventEvent);
      procedure SetOnWindowManagerEvent(AValue: TOnWindowManagerEventEvent);
    protected
      //procedure Quit(const event: TSDL_QuitEvent); virtual; abstract;
      //procedure WindowEvent(const event: TSDL_WindowEvent); virtual; abstract;
      //procedure WindowManagerEvent(const event: TSDL_SysWMEvent); virtual; abstract;
      //procedure KeyDown(const event: TSDL_KeyboardEvent); virtual; abstract;
      //procedure KeyUp(const event: TSDL_KeyboardEvent); virtual; abstract;
      //procedure TextEditing(const event: TSDL_TextEditingEvent); virtual; abstract;
      //procedure TextInput(const event: TSDL_TextInputEvent); virtual; abstract;
      //procedure MouseMotion(const event: TSDL_MouseMotionEvent); virtual; abstract;
      //procedure MouseButtonDown(const event: TSDL_MouseButtonEvent); virtual; abstract;
      //procedure MouseButtonUp(const event: TSDL_MouseButtonEvent); virtual; abstract;
      //procedure MouseWheel(const event: TSDL_MouseWheelEvent); virtual; abstract;
      //procedure JoyAxisMotion(const event: TSDL_JoyAxisEvent); virtual; abstract;
      //procedure JoyBallMotion(const event: TSDL_JoyBallEvent); virtual; abstract;
      //procedure JoyHatMotion(const event: TSDL_JoyHatEvent); virtual; abstract;
      //procedure JoyButtonDown(const event: TSDL_JoyButtonEvent); virtual; abstract;
      //procedure JoyButtonUp(const event: TSDL_JoyButtonEvent); virtual; abstract;
      //procedure JoyDeviceAdded(const event: TSDL_JoyDeviceEvent); virtual; abstract;
      //procedure JoyDeviceRemoved(const event: TSDL_JoyDeviceEvent); virtual; abstract;
      //procedure ControllerAxisMotion(const event: TSDL_ControllerAxisEvent); virtual; abstract;
      //procedure ControllerButtonDown(const event: TSDL_ControllerButtonEvent); virtual; abstract;
      //procedure ControllerButtonUp(const event: TSDL_ControllerButtonEvent); virtual; abstract;
      //procedure ControllerDeviceAdded(const event: TSDL_ControllerDeviceEvent); virtual; abstract;
      //procedure ControllerDeviceRemoved(const event: TSDL_ControllerDeviceEvent); virtual; abstract;
      //procedure ControllerDeviceRemapped(const event: TSDL_ControllerDeviceEvent); virtual; abstract;
      //procedure UserEvent(const event: TSDL_UserEvent); virtual; abstract;
      function UserEventRegistered(AEvent: TSDL_EventType) : Boolean;
      property OnQuit : TOnQuitEvent read FOnQuit write SetOnQuit;
      property OnWindowEvent : TOnWindowEventEvent read FOnWindowEvent write SetOnWindowEvent;
      property OnWindowManagerEvent : TOnWindowManagerEventEvent read FOnWindowManagerEvent write SetOnWindowManagerEvent;
      property OnKeyDown : TOnKeyDownEvent read FOnKeyDown write SetOnKeyDown;
      property OnKeyUp : TOnKeyUpEvent read FOnKeyUp write SetOnKeyUp;
      property OnTextEditing : TOnTextEditingEvent read FOnTextEditing write SetOnTextEditing;
      property OnTextInput : TOnTextInputEvent read FOnTextInput write SetOnTextInput;
      property OnMouseMotion : TOnMouseMotionEvent read FOnMouseMotion write SetOnMouseMotion;
      property OnMouseButtonDown : TOnMouseButtonDownEvent read FOnMouseButtonDown write SetOnMouseButtonDown;
      property OnMouseButtonUp : TOnMouseButtonUpEvent read FOnMouseButtonUp write SetOnMouseButtonUp;
      property OnMouseWheel : TOnMouseWheelEvent read FOnMouseWheel write SetOnMouseWheel;
      property OnJoyAxisMotion : TOnJoyAxisMotionEvent read FOnJoyAxisMotion write SetOnJoyAxisMotion;
      property OnJoyBallMotion : TOnJoyBallMotionEvent read FOnJoyBallMotion write SetOnJoyBallMotion;
      property OnJoyHatMotion : TOnJoyHatMotionEvent read FOnJoyHatMotion write SetOnJoyHatMotion;
      property OnJoyButtonDown : TOnJoyButtonDownEvent read FOnJoyButtonDown write SetOnJoyButtonDown;
      property OnJoyButtonUp : TOnJoyButtonUpEvent read FOnJoyButtonUp write SetOnJoyButtonUp;
      property OnJoyDeviceAdded : TOnJoyDeviceAddedEvent read FOnJoyDeviceAdded write SetOnJoyDeviceAdded;
      property OnJoyDeviceRemoved : TOnJoyDeviceRemovedEvent read FOnJoyDeviceRemoved write SetOnJoyDeviceRemoved;
      property OnControllerAxisMotion : TOnControllerAxisMotionEvent read FOnControllerAxisMotion write SetOnControllerAxisMotion;
      property OnControllerButtonDown : TOnControllerButtonDownEvent read FOnControllerButtonDown write SetOnControllerButtonDown;
      property OnControllerButtonUp : TOnControllerButtonUpEvent read FOnControllerButtonUp write SetOnControllerButtonUp;
      property OnControllerDeviceAdded : TOnControllerDeviceAddedEvent read FOnControllerDeviceAdded write SetOnControllerDeviceAdded;
      property OnControllerDeviceRemoved : TOnControllerDeviceRemovedEvent read FOnControllerDeviceRemoved write SetOnControllerDeviceRemoved;
      property OnControllerDeviceRemapped : TOnControllerDeviceRemappedEvent read FOnControllerDeviceRemapped write SetOnControllerDeviceRemapped;
      property OnControllerTouchPadMotion : TOnControllerTouchPadMotionEvent read FOnControllerTouchPadMotion write SetOnControllerTouchPadMotion;
      property OnControllerSensorUpdate : TOnControllerSensorUpdateEvent read FOnControllerSensorUpdateEvent write SetOnControllerSensorUpdateEvent;
      property OnUserEvent : TOnUserEvent read FOnUserEvent write SetOnUserEvent;
    public
      procedure HandlePending;
      constructor Create;
  end;

  function GetShiftState: TCustomShiftState;
  function GetKeyState(KeyCode : TSDL_ScanCode; AKeyboardState: pcuint8) : Boolean;

const
  SDL_USEREVENTSTOREGISTER = 5;

implementation

const
  SDL_USEREVENT_HIGH = SDL_USEREVENT+SDL_USEREVENTSTOREGISTER;

function GetKeyState(KeyCode: TSDL_ScanCode; AKeyboardState: pcuint8): Boolean;
begin
  Result := AKeyboardState[KeyCode] <> 0;
end;

function GetShiftState: TCustomShiftState;
var
  KeyboardState: pcuint8 = nil;
begin
  Result := [];
  if not Assigned(KeyboardState) then
    KeyboardState := SDL_GetKeyboardState(nil);

  if GetKeyState(SDL_SCANCODE_LSHIFT, KeyboardState) then
    Include(Result, ssShift);

  if GetKeyState(SDL_SCANCODE_RSHIFT, KeyboardState) then
    Include(Result, ssShift);

  if GetKeyState(SDL_SCANCODE_LALT, KeyboardState) then
    Include(Result, ssAlt);

  if GetKeyState(SDL_SCANCODE_RALT, KeyboardState) then
    Include(Result, ssAlt);

  if GetKeyState(SDL_SCANCODE_LCTRL, KeyboardState) then
    Include(Result, ssCtrl);

  if GetKeyState(SDL_SCANCODE_RCTRL, KeyboardState) then
    Include(Result, ssCtrl);

  if GetKeyState(SDL_SCANCODE_SPACE, KeyboardState) then
    Include(Result, ssSpace);

  if GetKeyState(SDL_SCANCODE_UP, KeyboardState) then
    Include(Result, ssUp);

  if GetKeyState(SDL_SCANCODE_DOWN, KeyboardState) then
    Include(Result, ssDown);

  if GetKeyState(SDL_SCANCODE_LEFT, KeyboardState) then
    Include(Result, ssLeft);

  if GetKeyState(SDL_SCANCODE_RIGHT, KeyboardState) then
    Include(Result, ssRight);

  if GetKeyState(SDL_SCANCODE_RETURN2, KeyboardState) then
    Include(Result, ssEnter);
end;


procedure TEventHandler.HandleEvent(const event: TSDL_Event);
begin
  case event.type_ of
    SDL_QUITEV:
      if Assigned(OnQuit) then
        OnQuit(event.quit);
    SDL_WINDOWEVENT:
      if Assigned(OnWindowEvent) then
        OnWindowEvent(event.window);
    SDL_SYSWMEVENT:
      if Assigned(OnWindowManagerEvent) then
        OnWindowManagerEvent(event.syswm);
    SDL_KEYDOWN:
      if Assigned(OnKeyDown) then
        OnKeyDown(event.key);
    SDL_KEYUP:
      if Assigned(OnKeyUp) then
        OnKeyUp(event.key);
    SDL_TEXTEDITING:
      if Assigned(OnTextEditing) then
        OnTextEditing(event.edit);
    SDL_TEXTINPUT:
      if Assigned(OnTextInput) then
        OnTextInput(event.text);
    SDL_MOUSEMOTION:
      if Assigned(OnMouseMotion) then
        OnMouseMotion(event.motion);
    SDL_MOUSEBUTTONDOWN:
      if Assigned(OnMouseButtonDown) then
        OnMouseButtonDown(event.button);
    SDL_MOUSEBUTTONUP:
      if Assigned(OnMouseButtonUp) then
        OnMouseButtonUp(event.button);
    SDL_MOUSEWHEEL:
      if Assigned(OnMouseWheel) then
        OnMouseWheel(event.wheel);
    SDL_JOYAXISMOTION:
      if Assigned(OnJoyAxisMotion) then
        OnJoyAxisMotion(event.jaxis);
    SDL_JOYBALLMOTION:
      if Assigned(OnJoyBallMotion) then
        OnJoyBallMotion(event.jball);
    SDL_JOYHATMOTION:
      if Assigned(OnJoyHatMotion) then
        OnJoyHatMotion(event.jhat);
    SDL_JOYBUTTONDOWN:
      if Assigned(OnJoyButtonDown) then
        OnJoyButtonDown(event.jbutton);
    SDL_JOYBUTTONUP:
      if Assigned(OnJoyButtonUp) then
        OnJoyButtonUp(event.jbutton);
    SDL_JOYDEVICEADDED:
      if Assigned(OnJoyDeviceAdded) then
        OnJoyDeviceAdded(event.jdevice);
    SDL_JOYDEVICEREMOVED:
      if Assigned(OnJoyDeviceRemoved) then
        OnJoyDeviceRemoved(event.jdevice);
    SDL_CONTROLLERAXISMOTION:
      if Assigned(OnControllerAxisMotion) then
        OnControllerAxisMotion(event.caxis);
    SDL_CONTROLLERBUTTONDOWN:
      if Assigned(OnControllerButtonDown) then
        OnControllerButtonDown(event.cbutton);
    SDL_CONTROLLERBUTTONUP:
      if Assigned(OnControllerButtonUp) then
        OnControllerButtonUp(event.cbutton);
    SDL_CONTROLLERDEVICEADDED:
      if Assigned(OnControllerDeviceAdded) then
        OnControllerDeviceAdded(event.cdevice);
    SDL_CONTROLLERDEVICEREMOVED:
      if Assigned(OnControllerDeviceRemoved) then
        OnControllerDeviceRemoved(event.cdevice);
    SDL_CONTROLLERDEVICEREMAPPED:
      if Assigned(OnControllerDeviceRemapped) then
        OnControllerDeviceRemapped(event.cdevice);
    SDL_CONTROLLERTOUCHPADMOTION:
      if Assigned(OnControllerTouchPadMotion) then
        OnControllerTouchPadMotion(event.ctouchpad);
    SDL_CONTROLLERSENSORUPDATE:
      if Assigned(OnControllerSensorUpdate) then
        OnControllerSensorUpdate(event.csensor);
    SDL_USEREVENT..SDL_USEREVENT_HIGH:
      if Assigned(OnUserEvent) then
        OnUserEvent(event.user);
  end;
end;

procedure TEventHandler.SetOnControllerAxisMotion(
  AValue: TOnControllerAxisMotionEvent);
begin
  if FOnControllerAxisMotion=AValue then Exit;
  FOnControllerAxisMotion:=AValue;
end;

procedure TEventHandler.SetOnControllerButtonDown(
  AValue: TOnControllerButtonDownEvent);
begin
  if FOnControllerButtonDown=AValue then Exit;
  FOnControllerButtonDown:=AValue;
end;

procedure TEventHandler.SetOnControllerButtonUp(
  AValue: TOnControllerButtonUpEvent);
begin
  if FOnControllerButtonUp=AValue then Exit;
  FOnControllerButtonUp:=AValue;
end;

procedure TEventHandler.SetOnControllerDeviceAdded(
  AValue: TOnControllerDeviceAddedEvent);
begin
  if FOnControllerDeviceAdded=AValue then Exit;
  FOnControllerDeviceAdded:=AValue;
end;

procedure TEventHandler.SetOnControllerDeviceRemapped(
  AValue: TOnControllerDeviceRemappedEvent);
begin
  if FOnControllerDeviceRemapped=AValue then Exit;
  FOnControllerDeviceRemapped:=AValue;
end;

procedure TEventHandler.SetOnControllerDeviceRemoved(
  AValue: TOnControllerDeviceRemovedEvent);
begin
  if FOnControllerDeviceRemoved=AValue then Exit;
  FOnControllerDeviceRemoved:=AValue;
end;

procedure TEventHandler.SetOnControllerTouchPadMotion(
  AValue: TOnControllerTouchPadMotionEvent);
begin
  if FOnControllerTouchPadMotion = AValue then Exit;
  FOnControllerTouchPadMotion := AValue;
end;

procedure TEventHandler.SetOnJoyAxisMotion(AValue: TOnJoyAxisMotionEvent);
begin
  if FOnJoyAxisMotion=AValue then Exit;
  FOnJoyAxisMotion:=AValue;
end;

procedure TEventHandler.SetOnJoyBallMotion(AValue: TOnJoyBallMotionEvent);
begin
  if FOnJoyBallMotion=AValue then Exit;
  FOnJoyBallMotion:=AValue;
end;

procedure TEventHandler.SetOnJoyButtonDown(AValue: TOnJoyButtonDownEvent);
begin
  if FOnJoyButtonDown=AValue then Exit;
  FOnJoyButtonDown:=AValue;
end;

procedure TEventHandler.SetOnJoyButtonUp(AValue: TOnJoyButtonUpEvent);
begin
  if FOnJoyButtonUp=AValue then Exit;
  FOnJoyButtonUp:=AValue;
end;

procedure TEventHandler.SetOnJoyDeviceAdded(AValue: TOnJoyDeviceAddedEvent);
begin
  if FOnJoyDeviceAdded=AValue then Exit;
  FOnJoyDeviceAdded:=AValue;
end;

procedure TEventHandler.SetOnJoyDeviceRemoved(AValue: TOnJoyDeviceRemovedEvent);
begin
  if FOnJoyDeviceRemoved=AValue then Exit;
  FOnJoyDeviceRemoved:=AValue;
end;

procedure TEventHandler.SetOnJoyHatMotion(AValue: TOnJoyHatMotionEvent);
begin
  if FOnJoyHatMotion=AValue then Exit;
  FOnJoyHatMotion:=AValue;
end;

procedure TEventHandler.SetOnKeyDown(AValue: TOnKeyDownEvent);
begin
  if FOnKeyDown=AValue then Exit;
  FOnKeyDown:=AValue;
end;

procedure TEventHandler.SetOnKeyUp(AValue: TOnKeyUpEvent);
begin
  if FOnKeyUp=AValue then Exit;
  FOnKeyUp:=AValue;
end;

procedure TEventHandler.SetOnMouseButtonDown(AValue: TOnMouseButtonDownEvent);
begin
  if FOnMouseButtonDown=AValue then Exit;
  FOnMouseButtonDown:=AValue;
end;

procedure TEventHandler.SetOnMouseButtonUp(AValue: TOnMouseButtonUpEvent);
begin
  if FOnMouseButtonUp=AValue then Exit;
  FOnMouseButtonUp:=AValue;
end;

procedure TEventHandler.SetOnMouseMotion(AValue: TOnMouseMotionEvent);
begin
  if FOnMouseMotion=AValue then Exit;
  FOnMouseMotion:=AValue;
end;

procedure TEventHandler.SetOnMouseWheel(AValue: TOnMouseWheelEvent);
begin
  if FOnMouseWheel=AValue then Exit;
  FOnMouseWheel:=AValue;
end;

procedure TEventHandler.SetOnQuit(AValue: TOnQuitEvent);
begin
  if FOnQuit=AValue then Exit;
  FOnQuit:=AValue;
end;

procedure TEventHandler.SetOnControllerSensorUpdateEvent(AValue: TOnControllerSensorUpdateEvent);
begin
  if FOnControllerSensorUpdateEvent = AValue then Exit;
  FOnControllerSensorUpdateEvent := AValue;
end;

procedure TEventHandler.SetOnTextEditing(AValue: TOnTextEditingEvent);
begin
  if FOnTextEditing=AValue then Exit;
  FOnTextEditing:=AValue;
end;

procedure TEventHandler.SetOnTextInput(AValue: TOnTextInputEvent);
begin
  if FOnTextInput=AValue then Exit;
  FOnTextInput:=AValue;
end;

procedure TEventHandler.SetOnUserEvent(AValue: TOnUserEvent);
begin
  if FOnUserEvent=AValue then Exit;
  FOnUserEvent:=AValue;
end;

procedure TEventHandler.SetOnWindowEvent(AValue: TOnWindowEventEvent);
begin
  if FOnWindowEvent=AValue then Exit;
  FOnWindowEvent:=AValue;
end;

procedure TEventHandler.SetOnWindowManagerEvent(
  AValue: TOnWindowManagerEventEvent);
begin
  if FOnWindowManagerEvent=AValue then Exit;
  FOnWindowManagerEvent:=AValue;
end;

function TEventHandler.UserEventRegistered(AEvent: TSDL_EventType): Boolean;
begin
  Result := (AEvent >=SDL_USEREVENT) and (AEvent <=SDL_USEREVENT_HIGH);
end;

procedure TEventHandler.HandlePending;
var
  Event : TSDL_Event;
begin
  while SDL_PollEvent(@Event) <> 0 do
    HandleEvent(Event);
end;

constructor TEventHandler.Create;
begin
  inherited Create;
  SDL_RegisterEvents(SDL_USEREVENTSTOREGISTER);
  FOnControllerAxisMotion:= nil;
  FOnControllerButtonDown:= nil;
  FOnControllerButtonUp:= nil;
  FOnControllerDeviceAdded:= nil;
  FOnControllerDeviceRemapped:= nil;
  FOnControllerDeviceRemoved:= nil;
  FOnJoyAxisMotion:= nil;
  FOnJoyBallMotion:= nil;
  FOnJoyButtonDown:= nil;
  FOnJoyButtonUp:= nil;
  FOnJoyDeviceAdded:= nil;
  FOnJoyDeviceRemoved:= nil;
  FOnJoyHatMotion:= nil;
  FOnKeyDown:= nil;
  FOnKeyUp:= nil;
  FOnMouseButtonDown:= nil;
  FOnMouseButtonUp:= nil;
  FOnMouseMotion:= nil;
  FOnMouseWheel:= nil;
  FOnQuit:= nil;
  FOnTextEditing:= nil;
  FOnTextInput:= nil;
  FOnUserEvent:= nil;
  FOnWindowEvent:= nil;
  FOnWindowManagerEvent:= nil;
end;


end.

