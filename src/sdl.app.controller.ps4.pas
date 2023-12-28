unit sdl.app.controller.ps4;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2,
  sdl.app.events.abstract,
  sdl.app.controller;

type

  { TSDLPS4Controller }

  TSDLPS4Controller = class(TController)
  private
    FWidth : integer;
    FHeight: integer;
    FGameController : PSDL_GameController;
    FOnControllerAxisMotion: TOnControllerAxisMotionEvent;
    FOnControllerButtonDown: TOnControllerButtonDownEvent;
    FOnControllerButtonUp: TOnControllerButtonUpEvent;
    procedure ControllerAxisMotion(const event: TSDL_ControllerAxisEvent);
    procedure ControllerButtonDown(const event: TSDL_ControllerButtonEvent);
    procedure ControllerButtonUp(const event: TSDL_ControllerButtonEvent);
    procedure ControllerDeviceAdded(const event: TSDL_ControllerDeviceEvent);
    procedure ControllerDeviceRemoved(const event: TSDL_ControllerDeviceEvent);
    procedure SetOnControllerAxisMotion(AValue: TOnControllerAxisMotionEvent);
    procedure SetOnControllerButtonDown(AValue: TOnControllerButtonDownEvent);
    procedure SetOnControllerButtonUp(AValue: TOnControllerButtonUpEvent);
  public
    constructor Create; override;
    destructor Destroy; override;
    property OnControllerAxisMotion : TOnControllerAxisMotionEvent
      read FOnControllerAxisMotion write SetOnControllerAxisMotion;

    property OnControllerButtonDown : TOnControllerButtonDownEvent
      read FOnControllerButtonDown write SetOnControllerButtonDown;

    property OnControllerButtonUp : TOnControllerButtonUpEvent
      read FOnControllerButtonUp write SetOnControllerButtonUp;
  end;

implementation

uses
  sdl.app.output,
  sdl.app.events.custom,
  sdl.app.video.methods,
  timestamps.types,
  timestamps;

{ TSDLPS4Controller }

const
  PS4_BUTTON_SHARE = 4;
  PS4_BUTTON_HOME = 5;
  PS4_BUTTON_OPTIONS = 6;

var
  LastTimeStamp : TLargerFloat = 0;

procedure TSDLPS4Controller.ControllerAxisMotion(
  const event: TSDL_ControllerAxisEvent);
const
  LTimeDelta = 0.100;
  LSensitivity = 0.50;
  LMaxValue = 32767;
  LHumbleTime = 100;
  LHumblePower = $FFFF;

var
  LTimestamp : TLargerFloat;
  procedure SelectH(AValue : SmallInt);
  begin
    if AValue < 0 then begin
      FNavigator.SelectLeft;
    end else begin
      FNavigator.SelectRight;
    end;
  end;

  procedure SelectV(AValue : SmallInt);
  begin
    if AValue < 0 then begin
      FNavigator.SelectUp;
    end else begin
      FNavigator.SelectDown;
    end;
  end;

begin
  LTimestamp := ClockMonotonic;
  case event.axis of
    SDL_CONTROLLER_AXIS_LEFTY, SDL_CONTROLLER_AXIS_RIGHTY: begin
      if (LTimestamp - LastTimeStamp) > LTimeDelta then begin
        LastTimeStamp := LTimestamp;
        if Abs(event.value/LMaxValue) > LSensitivity then begin
          SelectV(event.value);
          SDL_GameControllerRumble(
            FGameController, LHumblePower, LHumblePower, LHumbleTime);
        end;
      end;
    end;

    SDL_CONTROLLER_AXIS_LEFTX, SDL_CONTROLLER_AXIS_RIGHTX: begin
      if (LTimestamp - LastTimeStamp) > LTimeDelta then begin
        LastTimeStamp := LTimestamp;
        if Abs(event.value/LMaxValue) > LSensitivity then begin
          SelectH(event.value);
          SDL_GameControllerRumble(
            FGameController, LHumblePower, LHumblePower, LHumbleTime);
        end;
      end;
    end;

    SDL_CONTROLLER_AXIS_TRIGGERLEFT:;
    SDL_CONTROLLER_AXIS_TRIGGERRIGHT:;

    otherwise begin

    end;
  end;
end;

procedure TSDLPS4Controller.ControllerButtonDown(
  const event: TSDL_ControllerButtonEvent);
begin
  Print('');
  Print('Button Event');
  Print(event.button.ToString);
  Print(IntToStr(event.which));
end;

procedure TSDLPS4Controller.ControllerButtonUp(
  const event: TSDL_ControllerButtonEvent);
begin
  case event.button of
    SDL_CONTROLLER_BUTTON_A: ;
    SDL_CONTROLLER_BUTTON_B: ;
    SDL_CONTROLLER_BUTTON_X: ;
    SDL_CONTROLLER_BUTTON_Y: ;

    PS4_BUTTON_SHARE: ;
    PS4_BUTTON_HOME: ;
    PS4_BUTTON_OPTIONS: ;

    SDL_CONTROLLER_BUTTON_LEFTSTICK: ;
    SDL_CONTROLLER_BUTTON_RIGHTSTICK: ;
    SDL_CONTROLLER_BUTTON_LEFTSHOULDER: FNavigator.ConfirmSelection;
    SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: FNavigator.ConfirmSelection;

    SDL_CONTROLLER_BUTTON_DPAD_UP: FNavigator.SelectUp;
    SDL_CONTROLLER_BUTTON_DPAD_DOWN: FNavigator.SelectDown;
    SDL_CONTROLLER_BUTTON_DPAD_LEFT: FNavigator.SelectLeft;
    SDL_CONTROLLER_BUTTON_DPAD_RIGHT: FNavigator.SelectRight;

    SDL_CONTROLLER_BUTTON_TOUCHPAD: ;
  end;
end;

procedure TSDLPS4Controller.ControllerDeviceAdded(
  const event: TSDL_ControllerDeviceEvent);
begin

end;

procedure TSDLPS4Controller.ControllerDeviceRemoved(
  const event: TSDL_ControllerDeviceEvent);
begin

end;

procedure TSDLPS4Controller.SetOnControllerAxisMotion(
  AValue: TOnControllerAxisMotionEvent);
begin
  if FOnControllerAxisMotion = AValue then Exit;
  FOnControllerAxisMotion := AValue;
end;

procedure TSDLPS4Controller.SetOnControllerButtonDown(
  AValue: TOnControllerButtonDownEvent);
begin
  if FOnControllerButtonDown = AValue then Exit;
  FOnControllerButtonDown := AValue;
end;

procedure TSDLPS4Controller.SetOnControllerButtonUp(
  AValue: TOnControllerButtonUpEvent);
begin
  if FOnControllerButtonUp = AValue then Exit;
  FOnControllerButtonUp := AValue;
end;

constructor TSDLPS4Controller.Create;
var
  LGameControllers : integer;
  LWindowRect : TSDL_Rect;
begin
  inherited Create;
  LWindowRect := WindowSize;
  FWidth := LWindowRect.w;
  FHeight := LWindowRect.h;

  SDLEvents.OnControllerButtonDown := @ControllerButtonDown;
  SDLEvents.OnControllerAxisMotion := @ControllerAxisMotion;
  SDLEvents.OnControllerButtonUp := @ControllerButtonUp;

  SDL_JoystickUpdate;
  LGameControllers := SDL_NumJoysticks;
  if LGameControllers > 0 then begin
    // 'PS4 Controller'
    Print(SDL_GameControllerNameForIndex(0));
    FGameController := SDL_GameControllerOpen(0);
  end else begin

  end;
end;

destructor TSDLPS4Controller.Destroy;
begin
  SDL_GameControllerClose(FGameController);
  inherited Destroy;
end;

end.


